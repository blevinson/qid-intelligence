"""Graphiti LLMClient backed by Claude Code subscription.

Uses the local `claude` CLI via the Claude Agent SDK so requests are
authenticated against the user's OAuth credentials (in /data/.claude-home
in-pod) and billed against the Pro/Max subscription instead of an
Anthropic API key. Same pattern tradefarm uses in
src/tradefarm/agents/llm_providers.py::ClaudeCodeProvider.

This sits next to graphiti-core's existing AnthropicClient — the only
substantive difference is that we go through the CLI subprocess instead
of httpx-to-api.anthropic.com, which means no per-token billing.

Concurrency is bounded by a module-level asyncio semaphore: each call
spawns a `claude` subprocess and we don't want crucix-bridge / alpaca-
bridge fan-out forking ten of them at once. Default 4 concurrent calls
matches tradefarm.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import typing
from json import JSONDecodeError

from pydantic import BaseModel, ValidationError

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RefusalError
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"  # entity extraction is high-frequency; Haiku is the right choice
DEFAULT_SMALL_MODEL = "claude-haiku-4-5"
DEFAULT_CONCURRENCY = int(os.environ.get("CLAUDE_CODE_CONCURRENCY", "4"))
CLAUDE_CLI_TIMEOUT = int(os.environ.get("CLAUDE_CLI_TIMEOUT", "90"))

_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(DEFAULT_CONCURRENCY)
    return _sem


_FENCE_RE = re.compile(r"^```(?:json|javascript|js)?\n?|\n?```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    """Claude often wraps JSON in ```json ... ``` even when asked not to."""
    s = text.strip()
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s).strip()
    return s


def _extract_first_json_object(text: str) -> str:
    """Find the first balanced { ... } block. Used as a fallback when the
    model adds prose before/after the JSON despite instructions."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return text  # no balanced object found


class ClaudeCodeLLMClient(LLMClient):
    """Graphiti LLMClient that delegates to `claude` CLI via the Agent SDK.

    Args:
        config: Standard graphiti LLMConfig. `model` and `small_model` accept
            either Claude model aliases ("haiku", "sonnet") or full model IDs
            ("claude-haiku-4-5-latest"). Defaults to claude-haiku-4-5 for both.
        cache: Forwarded to LLMClient base class.
    """

    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        if config is None:
            config = LLMConfig()
        if config.model is None:
            config.model = DEFAULT_MODEL
        if config.small_model is None:
            config.small_model = DEFAULT_SMALL_MODEL
        super().__init__(config, cache)

    def _get_provider_type(self) -> str:
        return "anthropic"

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,  # noqa: ARG002 — claude CLI doesn't expose a knob
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        # We invoke `claude -p --output-format json` as a subprocess rather than
        # the Agent SDK's streaming query(): the SDK's control-protocol
        # `initialize` handshake hangs in this container, while simple print mode
        # is verified working with the in-pod subscription creds (HOME=/data/
        # .claude-home). Concurrency is bounded so a batch doesn't fork N CLIs.
        model = self.small_model if model_size == ModelSize.small else self.model
        if not model:
            model = DEFAULT_MODEL

        # Graphiti's convention: messages[0] is the system prompt; messages[1:]
        # are user/assistant turns. Collapse to a CLI system prompt + piped user
        # prompt.
        if not messages:
            raise ValueError("ClaudeCodeLLMClient: no messages to send")

        system_prompt = messages[0].content if messages[0].role == "system" else ""
        body_messages = messages[1:] if messages[0].role == "system" else messages

        user_parts: list[str] = []
        for m in body_messages:
            label = m.role.upper() if m.role else "USER"
            user_parts.append(f"[{label}]\n{m.content}")
        user_prompt = "\n\n".join(user_parts).strip()

        cmd = ["claude", "-p", "--output-format", "json", "--model", model]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]

        sem = _get_sem()
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out_b, err_b = await asyncio.wait_for(
                    proc.communicate(user_prompt.encode()),
                    timeout=CLAUDE_CLI_TIMEOUT,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error("ClaudeCodeLLMClient: claude CLI timed out after %ss", CLAUDE_CLI_TIMEOUT)
                raise

        if proc.returncode != 0:
            raise RefusalError(
                f"claude CLI exited {proc.returncode}: {err_b.decode(errors='replace')[:300]}"
            )

        # `-p --output-format json` returns an envelope; the model's text answer
        # is in `.result`. Fall back to raw stdout if it isn't enveloped.
        out_s = out_b.decode(errors="replace").strip()
        try:
            envelope = json.loads(out_s)
            if isinstance(envelope, dict):
                if envelope.get("is_error"):
                    raise RefusalError(f"claude CLI error: {str(envelope.get('result'))[:300]}")
                raw = (envelope.get("result") or "").strip()
            else:
                raw = out_s
        except JSONDecodeError:
            raw = out_s

        if not raw:
            raise RefusalError("ClaudeCodeLLMClient: empty response from claude CLI")

        # Strip code fences, then try direct JSON parse, then fall back to
        # extracting the first balanced JSON object.
        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except JSONDecodeError:
            extracted = _extract_first_json_object(cleaned)
            try:
                parsed = json.loads(extracted)
            except JSONDecodeError as e:
                logger.error(
                    "ClaudeCodeLLMClient: failed to parse JSON. Raw (truncated): %s",
                    raw[:500],
                )
                raise

        if not isinstance(parsed, dict):
            # Some prompts ask for a list (e.g. dedup). Wrap so the caller
            # can detect; graphiti's prompts always expect a dict, so this
            # is mostly defensive.
            parsed = {"value": parsed}

        # Validate against response_model if one was supplied. Graphiti's
        # base class injects the schema into the prompt; here we just check
        # the model accepts.
        if response_model is not None:
            try:
                response_model.model_validate(parsed)
            except ValidationError as e:
                logger.warning(
                    "ClaudeCodeLLMClient: response failed schema validation: %s. "
                    "Returning raw dict; graphiti will surface the error if it cares.",
                    e,
                )

        return parsed

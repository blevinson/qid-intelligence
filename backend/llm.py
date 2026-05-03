"""LLM adapters for crucix-bridge graphiti initialization.

Re-exports ClaudeCodeLLMClient, NoopCrossEncoder, TEIEmbedder — sourced from
the tradefarm _llm_adapters package and bundled here so Dockerfile.bridge can
COPY a single file into the image without dragging in the full tradefarm src.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import typing
from collections.abc import Iterable
from json import JSONDecodeError

import httpx
from pydantic import BaseModel, Field, ValidationError

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RefusalError
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ClaudeCodeLLMClient
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_SMALL_MODEL = "claude-haiku-4-5"
DEFAULT_CONCURRENCY = int(os.environ.get("CLAUDE_CODE_CONCURRENCY", "4"))

_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(DEFAULT_CONCURRENCY)
    return _sem


_FENCE_RE = re.compile(r"^```(?:json|javascript|js)?\n?|\n?```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s).strip()
    return s


def _extract_first_json_object(text: str) -> str:
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
    return text


class ClaudeCodeLLMClient(LLMClient):
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
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        from claude_agent_sdk import (  # type: ignore[import-not-found]
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )

        model = self.small_model if model_size == ModelSize.small else self.model
        if not model:
            model = DEFAULT_MODEL

        if not messages:
            raise ValueError("ClaudeCodeLLMClient: no messages to send")

        system_prompt = messages[0].content if messages[0].role == "system" else ""
        body_messages = messages[1:] if messages[0].role == "system" else messages

        user_parts: list[str] = []
        for m in body_messages:
            label = m.role.upper() if m.role else "USER"
            user_parts.append(f"[{label}]\n{m.content}")
        user_prompt = "\n\n".join(user_parts).strip()

        options = ClaudeAgentOptions(
            system_prompt=system_prompt or None,
            model=model,
            max_turns=1,
            allowed_tools=[],
            disallowed_tools=[],
            permission_mode="default",
            include_partial_messages=False,
        )

        sem = _get_sem()
        async with sem:
            chunks: list[str] = []
            try:
                async for msg in query(prompt=user_prompt, options=options):
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                chunks.append(block.text)
            except Exception as e:
                logger.error("ClaudeCodeLLMClient query failed: %s", e)
                raise

        raw = "".join(chunks).strip()
        if not raw:
            raise RefusalError("ClaudeCodeLLMClient: empty response from claude CLI")

        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except JSONDecodeError:
            extracted = _extract_first_json_object(cleaned)
            try:
                parsed = json.loads(extracted)
            except JSONDecodeError:
                logger.error("ClaudeCodeLLMClient: failed to parse JSON. Raw: %s", raw[:500])
                raise

        if not isinstance(parsed, dict):
            parsed = {"value": parsed}

        if response_model is not None:
            try:
                response_model.model_validate(parsed)
            except ValidationError as e:
                logger.warning("ClaudeCodeLLMClient: response failed schema validation: %s", e)

        return parsed


# ---------------------------------------------------------------------------
# NoopCrossEncoder
# ---------------------------------------------------------------------------


class NoopCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 0.0) for p in passages]


# ---------------------------------------------------------------------------
# TEIEmbedder
# ---------------------------------------------------------------------------


class TEIEmbedderConfig(BaseModel):
    api_url: str = Field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_API_URL",
            "http://tei-embed.cloudtorch.svc.cluster.local:8080/v1/embeddings",
        )
    )
    model: str = Field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_dim: int = Field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))
    )
    api_key: str | None = Field(default_factory=lambda: os.environ.get("EMBEDDING_API_KEY") or None)
    timeout_s: float = Field(default=30.0)


class TEIEmbedder(EmbedderClient):
    def __init__(self, config: TEIEmbedderConfig | None = None):
        self.config = config or TEIEmbedderConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(headers=headers, timeout=self.config.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        client = await self._get_client()
        body = {"model": self.config.model, "input": inputs}
        try:
            resp = await client.post(self.config.api_url, json=body)
        except httpx.HTTPError as e:
            logger.error("TEIEmbedder fetch failed: %s", e)
            raise

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", "2") or "2")
            logger.warning("TEIEmbedder rate-limited; sleeping %ss", retry_after)
            await asyncio.sleep(retry_after)
            resp = await client.post(self.config.api_url, json=body)

        if resp.status_code >= 400:
            raise RuntimeError(f"TEI embeddings {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        items = data.get("data", [])
        out: list[list[float]] = []
        for entry in items:
            emb = entry.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError(f"TEI: bad embedding payload: {entry!r}")
            if len(emb) != self.config.embedding_dim:
                raise RuntimeError(
                    f"TEI: expected dim {self.config.embedding_dim}, got {len(emb)}."
                )
            out.append([float(x) for x in emb])
        if len(out) != len(inputs):
            raise RuntimeError(f"TEI: requested {len(inputs)} embeddings, got {len(out)}")
        return out

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            texts = list(input_data)
        else:
            texts = [str(input_data)]
        embeddings = await self._embed(texts)
        return embeddings[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await self._embed(input_data_list)


__all__ = ["ClaudeCodeLLMClient", "NoopCrossEncoder", "TEIEmbedder"]

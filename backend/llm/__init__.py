"""LLM and embedder adapters for the Graphiti pipeline.

Local package used by crucix_bridge.py to swap Graphiti's default OpenAI
LLM/embedder/reranker for a Claude (Anthropic-protocol) + in-cluster TEI
stack. The same adapters are vendored into tradefarm/memory/_llm_adapters;
keep the two in sync until/unless we package them.

Provides:
- ClaudeCodeLLMClient: Graphiti LLMClient backed by the local `claude` CLI
  via the Claude Agent SDK. Subscription auth, no API key, no Azure.
- TEIEmbedder: Graphiti EmbedderClient pointing at the in-cluster TEI
  service (BGE-M3, 1024-dim) over its OpenAI-compatible /v1/embeddings.
"""

from .claude_code_llm_client import ClaudeCodeLLMClient
from .noop_cross_encoder import NoopCrossEncoder
from .tei_embedder import TEIEmbedder, TEIEmbedderConfig

__all__ = [
    "ClaudeCodeLLMClient",
    "NoopCrossEncoder",
    "TEIEmbedder",
    "TEIEmbedderConfig",
]

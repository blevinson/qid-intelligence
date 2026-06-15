"""No-op CrossEncoderClient for Graphiti.

Graphiti.__init__ defaults `cross_encoder` to OpenAIRerankerClient, which
fails to instantiate without OPENAI_API_KEY. We don't use search reranking
in the qid bridge (we only ingest episodes, never retrieve via Graphiti's
search), so a passthrough that returns equal scores is sufficient.

If/when reranking matters here, swap in a Claude-based ranker.
"""

from __future__ import annotations

from graphiti_core.cross_encoder.client import CrossEncoderClient


class NoopCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        # Preserve input order, give every passage the same neutral score.
        return [(p, 0.0) for p in passages]

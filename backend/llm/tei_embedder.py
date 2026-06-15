"""Graphiti EmbedderClient pointing at the in-cluster TEI service.

TEI (HuggingFace's text-embeddings-inference) exposes an OpenAI-compatible
/v1/embeddings endpoint. We could use Graphiti's OpenAIGenericEmbedder, but
that drags openai SDK init quirks (api_key required even when unused, etc.)
into the bridge. A small httpx client is simpler and matches the existing
TEI deployment we use for cloudtorch's pgvector pipeline.

Config defaults match the in-cluster TEI service deployed for cloudtorch:
- URL: http://tei-embed.cloudtorch.svc.cluster.local:8080/v1/embeddings
- Model: BAAI/bge-m3
- Dim: 1024 (matches existing graphiti name_embedding / fact_embedding)

Override via env (EMBEDDING_API_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
or by passing TEIEmbedderConfig explicitly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable

import httpx
from pydantic import BaseModel, Field

from graphiti_core.embedder.client import EmbedderClient

logger = logging.getLogger(__name__)


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
    """Async embedder hitting an OpenAI-compatible /v1/embeddings endpoint.

    BGE-M3 produces 1024-dim vectors, which matches what's already stored on
    the existing 13k+ Entity.name_embedding / RELATES_TO.fact_embedding rows
    in the qid neo4j graph (after backfill).
    """

    def __init__(self, config: TEIEmbedderConfig | None = None):
        self.config = config or TEIEmbedderConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=self.config.timeout_s,
            )
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
            text = resp.text[:500]
            raise RuntimeError(f"TEI embeddings {resp.status_code}: {text}")

        data = resp.json()
        items = data.get("data", [])
        out: list[list[float]] = []
        for entry in items:
            emb = entry.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError(f"TEI: bad embedding payload: {entry!r}")
            if len(emb) != self.config.embedding_dim:
                raise RuntimeError(
                    f"TEI: expected dim {self.config.embedding_dim}, got {len(emb)}. "
                    f"Update EMBEDDING_DIMENSIONS or rebuild the index."
                )
            out.append([float(x) for x in emb])
        if len(out) != len(inputs):
            raise RuntimeError(
                f"TEI: requested {len(inputs)} embeddings, got {len(out)}"
            )
        return out

    # ------- Graphiti EmbedderClient interface -------

    async def create(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[float]:
        # Graphiti uses this for one-text-at-a-time embedding (entity name,
        # edge fact). The signature is broader for OpenAI compat (token IDs)
        # but in practice we receive str or list[str].
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            texts = list(input_data)
        else:
            # token-id paths aren't used by graphiti against TEI; coerce.
            texts = [str(input_data)]

        embeddings = await self._embed(texts)
        # Match Graphiti's openai.py convention: when called with a single
        # string, return the raw vector; when called with a list, also
        # return the first vector (the openai client truncates to embedding_dim
        # and only returns the first one — see embedder/openai.py:60).
        return embeddings[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await self._embed(input_data_list)

"""
Graph builder service using Graphiti (local Neo4j).
Replaces the original Zep Cloud implementation.
"""

import asyncio
import json
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor
from ..utils.locale import t, get_locale, set_locale


@dataclass
class GraphInfo:
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    Graph builder using Graphiti + local Neo4j.
    Each 'graph' is a Graphiti group_id partition.
    """

    def __init__(self):
        self.task_manager = TaskManager()
        self._graphiti: Optional[Graphiti] = None

    def _get_graphiti(self) -> Graphiti:
        if self._graphiti is None:
            from graphiti_core.llm_client.openai_client import OpenAIClient
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

            llm_config = LLMConfig(
                api_key=Config.LLM_API_KEY or None,
                model=Config.LLM_MODEL_NAME or "gpt-5.4-mini",
                small_model=Config.LLM_MODEL_NAME or "gpt-5.4-mini",
                base_url=Config.LLM_BASE_URL or None,
            )
            embedder_config = OpenAIEmbedderConfig(
                api_key=Config.LLM_API_KEY or None,
                base_url=Config.LLM_BASE_URL or None,
            )
            self._graphiti = Graphiti(
                Config.NEO4J_URI,
                Config.NEO4J_USER,
                Config.NEO4J_PASSWORD,
                llm_client=OpenAIClient(config=llm_config),
                embedder=OpenAIEmbedder(config=embedder_config),
            )
        return self._graphiti

    # --- Compatibility methods for MiroFish API endpoints ---

    def create_graph(self, name: str = "default") -> str:
        """Create a graph partition. Returns group_id."""
        return Config.GRAPHITI_GROUP_ID

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """Ontology is advisory in Graphiti -- store for reference only."""
        pass

    def add_text_chunks(
        self,
        graph_id: str,
        chunks: List[str],
        progress_callback: Optional[Callable] = None,
    ):
        """Ingest text chunks as Graphiti episodes."""
        loop = asyncio.new_event_loop()
        graphiti = self._get_graphiti()
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            loop.run_until_complete(
                graphiti.add_episode(
                    name=f"chunk_{i}",
                    episode_body=chunk,
                    source=EpisodeType.text,
                    source_description=f"Text chunk {i+1}/{total}",
                    reference_time=datetime.now(timezone.utc),
                    group_id=graph_id,
                )
            )
            if progress_callback:
                progress_callback(f"Ingested chunk {i+1}/{total}", (i + 1) / total)
        loop.close()

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """Ingest text chunks as Graphiti episodes. Returns fake episode UUIDs."""
        self.add_text_chunks(graph_id, chunks, progress_callback)
        return [f"ep_{i}" for i in range(len(chunks))]

    def _wait_for_episodes(self, episode_uuids: List[str], progress_callback: Optional[Callable] = None):
        """Graphiti processes episodes synchronously, so nothing to wait for."""
        if progress_callback:
            progress_callback("Episodes processed", 1.0)

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """Get graph data for visualization."""
        loop = asyncio.new_event_loop()
        graphiti = self._get_graphiti()
        try:
            info = loop.run_until_complete(self._get_graph_info(graphiti, graph_id))
            # Query nodes and edges for the visualization
            nodes = []
            edges = []
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    Config.NEO4J_URI,
                    auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
                )
                with driver.session() as session:
                    result = session.run(
                        "MATCH (n:Entity) RETURN elementId(n) AS id, n.name AS name, "
                        "n.summary AS summary, labels(n) AS labels LIMIT 200"
                    )
                    for r in result:
                        nodes.append({
                            "id": r["id"],
                            "name": r["name"],
                            "summary": r["summary"],
                            "type": next((l for l in r["labels"] if l != "Entity"), "Entity"),
                        })
                    result = session.run(
                        "MATCH (a:Entity)-[r]->(b:Entity) "
                        "RETURN elementId(a) AS source, elementId(b) AS target, "
                        "type(r) AS type, r.fact AS fact LIMIT 500"
                    )
                    for r in result:
                        edges.append({
                            "source": r["source"],
                            "target": r["target"],
                            "type": r["type"],
                            "fact": r["fact"],
                        })
                driver.close()
            except Exception:
                pass

            return {
                "nodes": nodes,
                "edges": edges,
                "node_count": info.node_count,
                "edge_count": info.edge_count,
                "entity_types": info.entity_types,
            }
        finally:
            loop.close()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "QID Intelligence Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        current_locale = get_locale()
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, current_locale),
        )
        thread.daemon = True
        thread.start()
        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = "en",
    ):
        set_locale(locale)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._build_graph_async(
                    task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size
                )
            )
        except Exception as e:
            import traceback
            self.task_manager.fail_task(task_id, f"{e}\n{traceback.format_exc()}")
        finally:
            loop.close()

    async def _build_graph_async(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ):
        graphiti = self._get_graphiti()
        group_id = Config.GRAPHITI_GROUP_ID

        self.task_manager.update_task(
            task_id, status=TaskStatus.PROCESSING, progress=5,
            message=t("progress.startBuildingGraph"),
        )

        # Split text into chunks
        chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
        total_chunks = len(chunks)
        self.task_manager.update_task(
            task_id, progress=15,
            message=t("progress.textSplit", count=total_chunks),
        )

        # Ingest chunks as episodes
        for i, chunk in enumerate(chunks):
            await graphiti.add_episode(
                name=f"{graph_name}_chunk_{i}",
                episode_body=chunk,
                source=EpisodeType.text,
                source_description=f"Text chunk {i+1}/{total_chunks} from {graph_name}",
                reference_time=datetime.now(timezone.utc),
                group_id=group_id,
            )
            progress = 15 + int((i + 1) / total_chunks * 75)
            self.task_manager.update_task(
                task_id, progress=progress,
                message=f"Ingested chunk {i+1}/{total_chunks}",
            )

        # Get graph stats
        self.task_manager.update_task(
            task_id, progress=95, message="Fetching graph info...",
        )
        graph_info = await self._get_graph_info(graphiti, group_id)

        self.task_manager.complete_task(task_id, {
            "graph_id": group_id,
            "graph_info": graph_info.to_dict(),
            "chunks_processed": total_chunks,
        })

    async def add_episode(
        self,
        content: str,
        source_description: str,
        name: Optional[str] = None,
        episode_type: EpisodeType = EpisodeType.text,
        reference_time: Optional[datetime] = None,
        group_id: Optional[str] = None,
    ):
        """Add a single episode to the knowledge graph."""
        graphiti = self._get_graphiti()
        gid = group_id or Config.GRAPHITI_GROUP_ID
        ref_time = reference_time or datetime.now(timezone.utc)
        ep_name = name or f"episode_{ref_time.strftime('%Y%m%d_%H%M%S')}"

        if isinstance(content, dict):
            content = json.dumps(content)
            episode_type = EpisodeType.json

        return await graphiti.add_episode(
            name=ep_name,
            episode_body=content,
            source=episode_type,
            source_description=source_description,
            reference_time=ref_time,
            group_id=gid,
        )

    async def search(
        self,
        query: str,
        num_results: int = 10,
        group_id: Optional[str] = None,
    ):
        """Search the knowledge graph."""
        graphiti = self._get_graphiti()
        gid = group_id or Config.GRAPHITI_GROUP_ID
        return await graphiti.search(
            query=query,
            num_results=num_results,
            group_ids=[gid],
        )

    async def _get_graph_info(self, graphiti: Graphiti, group_id: str) -> GraphInfo:
        """Get basic graph statistics by querying Neo4j directly."""
        driver = graphiti.driver
        node_count = 0
        edge_count = 0
        entity_types = []

        try:
            records = await driver.execute_query(
                "MATCH (n) RETURN count(n) as cnt, collect(distinct labels(n)) as label_sets"
            )
            if records:
                node_count = records[0]["cnt"]
                for label_set in records[0].get("label_sets", []):
                    for label in label_set:
                        if label not in ("Entity", "Node", "Episodic"):
                            entity_types.append(label)
                entity_types = list(set(entity_types))

            edge_records = await driver.execute_query(
                "MATCH ()-[r]->() RETURN count(r) as cnt"
            )
            if edge_records:
                edge_count = edge_records[0]["cnt"]
        except Exception:
            pass

        return GraphInfo(
            graph_id=group_id,
            node_count=node_count,
            edge_count=edge_count,
            entity_types=entity_types,
        )

    async def close(self):
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None

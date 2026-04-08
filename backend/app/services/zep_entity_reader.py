"""
Entity reader and filter service using Graphiti (local Neo4j).
Replaces the original Zep Cloud implementation.
Preserves class names and interface for backward compatibility.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from graphiti_core import Graphiti

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.zep_entity_reader')


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


@dataclass
class EntityNode:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        for label in self.labels:
            if label not in ("Entity", "Node"):
                return label
        return None


@dataclass
class FilteredEntities:
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class ZepEntityReader:
    """
    Entity reader using sync Neo4j driver.
    Class name preserved for backward compatibility.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            )
        return self._driver

    def _query(self, cypher: str, **params) -> List[Dict[str, Any]]:
        driver = self._get_driver()
        try:
            with driver.session() as session:
                result = session.run(cypher, **params)
                return [dict(r) for r in result]
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            return []

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        logger.info(f"Fetching all nodes for graph {graph_id}")
        records = self._query(
            "MATCH (n:Entity) RETURN n.uuid as uuid, n.name as name, "
            "labels(n) as labels, n.summary as summary LIMIT 2000"
        )
        nodes_data = []
        for r in records:
            nodes_data.append({
                "uuid": r.get("uuid", ""),
                "name": r.get("name", ""),
                "labels": r.get("labels", []),
                "summary": r.get("summary", ""),
                "attributes": {},
            })
        logger.info(f"Fetched {len(nodes_data)} nodes")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        logger.info(f"Fetching all edges for graph {graph_id}")
        records = self._query(
            "MATCH (s:Entity)-[r]->(t:Entity) "
            "RETURN r.uuid as uuid, type(r) as name, r.fact as fact, "
            "s.uuid as source_uuid, t.uuid as target_uuid LIMIT 5000"
        )
        edges_data = []
        for r in records:
            edges_data.append({
                "uuid": r.get("uuid", ""),
                "name": r.get("name", ""),
                "fact": r.get("fact", ""),
                "source_node_uuid": r.get("source_uuid", ""),
                "target_node_uuid": r.get("target_uuid", ""),
                "attributes": {},
            })
        logger.info(f"Fetched {len(edges_data)} edges")
        return edges_data

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        records = self._query(
            "MATCH (n {uuid: $uuid})-[r]-(m) "
            "RETURN r.uuid as uuid, type(r) as name, r.fact as fact, "
            "n.uuid as source_uuid, m.uuid as target_uuid",
            uuid=node_uuid,
        )
        return [
            {
                "uuid": r.get("uuid", ""),
                "name": r.get("name", ""),
                "fact": r.get("fact", ""),
                "source_node_uuid": r.get("source_uuid", ""),
                "target_node_uuid": r.get("target_uuid", ""),
                "attributes": {},
            }
            for r in records
        ]

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """Filter nodes to those with custom labels (not just 'Entity'/'Node')."""
        logger.info(f"Filtering entities for graph {graph_id}")

        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities = []
        entity_types_found = set()

        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [l for l in labels if l not in ("Entity", "Node")]

            if not custom_labels:
                continue

            if defined_entity_types:
                matching = [l for l in custom_labels if l in defined_entity_types]
                if not matching:
                    continue
                entity_type = matching[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node.get("attributes", {}),
            )

            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges
                entity.related_nodes = [
                    {
                        "uuid": node_map[uid]["uuid"],
                        "name": node_map[uid]["name"],
                        "labels": node_map[uid]["labels"],
                        "summary": node_map[uid].get("summary", ""),
                    }
                    for uid in related_node_uuids
                    if uid in node_map
                ]

            filtered_entities.append(entity)

        logger.info(
            f"Filtered: {total_count} total -> {len(filtered_entities)} entities, "
            f"types: {entity_types_found}"
        )

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str,
    ) -> Optional[EntityNode]:
        """Get a single entity with all its edges and related nodes."""
        try:
            records = self._query(
                "MATCH (n {uuid: $uuid}) "
                "RETURN n.uuid as uuid, n.name as name, labels(n) as labels, "
                "n.summary as summary",
                uuid=entity_uuid,
            )
            if not records:
                return None

            r = records[0]
            edges = self.get_node_edges(entity_uuid)
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}

            related_edges = []
            related_node_uuids = set()

            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            return EntityNode(
                uuid=r.get("uuid", ""),
                name=r.get("name", ""),
                labels=r.get("labels", []),
                summary=r.get("summary", ""),
                attributes={},
                related_edges=related_edges,
                related_nodes=[
                    {
                        "uuid": node_map[uid]["uuid"],
                        "name": node_map[uid]["name"],
                        "labels": node_map[uid]["labels"],
                        "summary": node_map[uid].get("summary", ""),
                    }
                    for uid in related_node_uuids
                    if uid in node_map
                ],
            )
        except Exception as e:
            logger.error(f"Failed to get entity {entity_uuid}: {e}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities

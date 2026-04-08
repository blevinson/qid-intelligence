"""
Knowledge graph retrieval tools using Graphiti (local Neo4j).
Replaces the original Zep Cloud implementation.

Core retrieval tools:
1. InsightForge - Deep insight retrieval with LLM sub-question decomposition
2. PanoramaSearch - Broad search including historical/expired content
3. QuickSearch - Fast simple search
4. InterviewAgents - OASIS agent interviews (unchanged, OASIS-specific)
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.locale import get_locale, t

logger = get_logger('mirofish.zep_tools')


def _run_async(coro):
    """Run an async coroutine from sync context."""
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
class SearchResult:
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} relevant items"]
        if self.facts:
            text_parts.append("\n### Relevant Facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
        }

    def to_text(self) -> str:
        entity_type = next((l for l in self.labels if l not in ("Entity", "Node")), "Unknown")
        return f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
        }

    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relation: {source} --[{self.name}]--> {target}\nFact: {self.fact}"
        if include_temporal:
            base_text += f"\nValid: {self.valid_at or 'unknown'} - {self.invalid_at or 'present'}"
            if self.expired_at:
                base_text += f" (expired: {self.expired_at})"
        return base_text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
        }

    def to_text(self) -> str:
        text_parts = [
            "## Deep Analysis",
            f"Query: {self.query}",
            f"Scenario: {self.simulation_requirement}",
            f"\n### Data Statistics",
            f"- Relevant facts: {self.total_facts}",
            f"- Entities: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}",
        ]
        if self.sub_queries:
            text_parts.append("\n### Sub-queries")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        if self.semantic_facts:
            text_parts.append('\n### Key Facts (cite these in report)')
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.entity_insights:
            text_parts.append("\n### Core Entities")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get("summary"):
                    text_parts.append(f'  Summary: "{entity["summary"]}"')
        if self.relationship_chains:
            text_parts.append("\n### Relationship Chains")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count,
        }

    def to_text(self) -> str:
        text_parts = [
            "## Panorama Search Results",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Nodes: {self.total_nodes}",
            f"- Edges: {self.total_edges}",
            f"- Active facts: {self.active_count}",
            f"- Historical facts: {self.historical_count}",
        ]
        if self.active_facts:
            text_parts.append("\n### Active Facts")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.historical_facts:
            text_parts.append("\n### Historical Facts")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.all_nodes:
            text_parts.append("\n### Entities")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ("Entity", "Node")), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes,
        }


@dataclass
class InterviewResult:
    interview_topic: str
    interview_questions: List[str] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    selection_reasoning: str = ""
    summary: str = ""
    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "interviews": [i.to_dict() for i in self.interviews],
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count,
        }

    def to_text(self) -> str:
        text_parts = [
            f"## Interview Report: {self.interview_topic}",
            f"Interviewed: {self.interviewed_count}/{self.total_agents} agents",
        ]
        if self.summary:
            text_parts.append(f"\n### Summary\n{self.summary}")
        for interview in self.interviews:
            text_parts.append(f"\n### {interview.agent_name} ({interview.agent_role})")
            text_parts.append(interview.response[:500])
            if interview.key_quotes:
                for q in interview.key_quotes:
                    text_parts.append(f'  > "{q}"')
        return "\n".join(text_parts)


class ZepToolsService:
    """
    Knowledge graph retrieval using Graphiti + local Neo4j.
    Preserves the same public interface for report_agent.py compatibility.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._graphiti: Optional[Graphiti] = None
        self._llm_client = llm_client
        self._group_id = Config.GRAPHITI_GROUP_ID
        logger.info("GraphitiToolsService initialized")

    def _get_graphiti(self) -> Graphiti:
        if self._graphiti is None:
            self._graphiti = Graphiti(
                Config.NEO4J_URI,
                Config.NEO4J_USER,
                Config.NEO4J_PASSWORD,
            )
        return self._graphiti

    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> SearchResult:
        """Hybrid search via Graphiti (semantic + BM25)."""
        logger.info(f"Graphiti search: graph={graph_id}, query={query[:50]}")

        graphiti = self._get_graphiti()
        edges = _run_async(
            graphiti.search(
                query=query,
                num_results=limit,
                group_ids=[self._group_id],
            )
        )

        facts = []
        edges_data = []
        for edge in edges:
            if hasattr(edge, "fact") and edge.fact:
                facts.append(edge.fact)
            edges_data.append({
                "uuid": getattr(edge, "uuid", ""),
                "name": getattr(edge, "name", ""),
                "fact": getattr(edge, "fact", ""),
                "source_node_uuid": getattr(edge, "source_node_uuid", ""),
                "target_node_uuid": getattr(edge, "target_node_uuid", ""),
            })

        logger.info(f"Search complete: {len(facts)} facts")
        return SearchResult(
            facts=facts,
            edges=edges_data,
            nodes=[],
            query=query,
            total_count=len(facts),
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """Get all nodes via Neo4j Cypher."""
        graphiti = self._get_graphiti()
        records = _run_async(
            graphiti.driver.execute_query(
                "MATCH (n:Entity) RETURN n.uuid as uuid, n.name as name, "
                "labels(n) as labels, n.summary as summary LIMIT 2000"
            )
        )
        result = []
        for r in (records or []):
            result.append(NodeInfo(
                uuid=r.get("uuid", ""),
                name=r.get("name", ""),
                labels=r.get("labels", []),
                summary=r.get("summary", ""),
                attributes={},
            ))
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """Get all edges via Neo4j Cypher."""
        graphiti = self._get_graphiti()
        records = _run_async(
            graphiti.driver.execute_query(
                "MATCH (s)-[r:RELATES_TO]->(t) "
                "RETURN r.uuid as uuid, r.name as name, r.fact as fact, "
                "s.uuid as source_uuid, t.uuid as target_uuid, "
                "s.name as source_name, t.name as target_name, "
                "r.created_at as created_at, r.valid_at as valid_at, "
                "r.invalid_at as invalid_at, r.expired_at as expired_at "
                "LIMIT 5000"
            )
        )
        result = []
        for r in (records or []):
            result.append(EdgeInfo(
                uuid=r.get("uuid", ""),
                name=r.get("name", ""),
                fact=r.get("fact", ""),
                source_node_uuid=r.get("source_uuid", ""),
                target_node_uuid=r.get("target_uuid", ""),
                source_node_name=r.get("source_name"),
                target_node_name=r.get("target_name"),
                created_at=str(r["created_at"]) if r.get("created_at") else None,
                valid_at=str(r["valid_at"]) if r.get("valid_at") else None,
                invalid_at=str(r["invalid_at"]) if r.get("invalid_at") else None,
                expired_at=str(r["expired_at"]) if r.get("expired_at") else None,
            ))
        return result

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        """Get entities filtered by label type."""
        all_nodes = self.get_all_nodes(graph_id)
        return [n for n in all_nodes if entity_type in n.labels]

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """Get entity details and its relationships."""
        graphiti = self._get_graphiti()
        records = _run_async(
            graphiti.driver.execute_query(
                "MATCH (n:Entity {name: $name}) "
                "OPTIONAL MATCH (n)-[r]-(m) "
                "RETURN n.name as name, n.summary as summary, labels(n) as labels, "
                "collect({relation: type(r), fact: r.fact, other: m.name}) as relations",
                {"name": entity_name},
            )
        )
        if not records:
            return {"name": entity_name, "summary": "", "relations": []}
        rec = records[0]
        return {
            "name": rec.get("name", entity_name),
            "summary": rec.get("summary", ""),
            "labels": rec.get("labels", []),
            "relations": rec.get("relations", []),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Get graph statistics."""
        graphiti = self._get_graphiti()
        node_count = 0
        edge_count = 0
        entity_types = []

        try:
            nr = _run_async(graphiti.driver.execute_query("MATCH (n) RETURN count(n) as cnt"))
            if nr:
                node_count = nr[0]["cnt"]
            er = _run_async(graphiti.driver.execute_query("MATCH ()-[r]->() RETURN count(r) as cnt"))
            if er:
                edge_count = er[0]["cnt"]
            tr = _run_async(graphiti.driver.execute_query(
                "MATCH (n) UNWIND labels(n) as lbl "
                "WITH DISTINCT lbl WHERE NOT lbl IN ['Entity','Node','Episodic'] "
                "RETURN collect(lbl) as types"
            ))
            if tr:
                entity_types = tr[0].get("types", [])
        except Exception as e:
            logger.warning(f"Failed to get graph stats: {e}")

        return {
            "graph_id": graph_id,
            "node_count": node_count,
            "edge_count": edge_count,
            "entity_types": entity_types,
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
    ) -> str:
        """Get simulation context by searching the graph."""
        result = self.search_graph(graph_id, simulation_requirement, limit=20)
        if result.facts:
            return "\n".join(result.facts[:15])
        return "No relevant context found in knowledge graph."

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """Deep insight retrieval with LLM sub-question decomposition."""
        logger.info(f"InsightForge: {query[:50]}")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[],
        )

        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries

        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            sr = self.search_graph(graph_id, sub_query, limit=15)
            for fact in sr.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            all_edges.extend(sr.edges)

        main_sr = self.search_graph(graph_id, query, limit=20)
        for fact in main_sr.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                for key in ("source_node_uuid", "target_node_uuid"):
                    uid = edge_data.get(key, "")
                    if uid:
                        entity_uuids.add(uid)

        entity_insights = []
        graphiti = self._get_graphiti()
        for uid in list(entity_uuids)[:50]:
            try:
                records = _run_async(
                    graphiti.driver.execute_query(
                        "MATCH (n {uuid: $uuid}) RETURN n.name as name, n.summary as summary, labels(n) as labels",
                        {"uuid": uid},
                    )
                )
                if records:
                    r = records[0]
                    labels = r.get("labels", [])
                    etype = next((l for l in labels if l not in ("Entity", "Node")), "Entity")
                    entity_insights.append({
                        "name": r.get("name", ""),
                        "type": etype,
                        "summary": r.get("summary", ""),
                    })
            except Exception:
                pass

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        chains = []
        for edge_data in all_edges[:30]:
            if isinstance(edge_data, dict):
                src = edge_data.get("source_node_uuid", "")[:8]
                tgt = edge_data.get("target_node_uuid", "")[:8]
                name = edge_data.get("name", "")
                fact = edge_data.get("fact", "")
                if name and fact:
                    chains.append(f"{src}.. --[{name}]--> {tgt}.. : {fact}")
        result.relationship_chains = chains
        result.total_relationships = len(chains)

        logger.info(f"InsightForge complete: {result.total_facts} facts, {result.total_entities} entities")
        return result

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50,
    ) -> PanoramaResult:
        """Broad search including historical/expired content."""
        logger.info(f"PanoramaSearch: {query[:50]}")

        result = PanoramaResult(query=query)

        all_nodes = self.get_all_nodes(graph_id)
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        node_map = {n.uuid: n for n in all_nodes}
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue
            is_historical = edge.is_expired or edge.is_invalid
            if is_historical:
                valid_at = edge.valid_at or "unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "unknown"
                historical_facts.append(f"[{valid_at} - {invalid_at}] {edge.fact}")
            else:
                active_facts.append(edge.fact)

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(",", " ").split() if len(w.strip()) > 1]

        def relevance_score(fact: str) -> int:
            fl = fact.lower()
            score = 100 if query_lower in fl else 0
            for kw in keywords:
                if kw in fl:
                    score += 10
            return score

        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(f"PanoramaSearch complete: {result.active_count} active, {result.historical_count} historical")
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> SearchResult:
        """Fast simple search."""
        logger.info(f"QuickSearch: {query[:50]}")
        result = self.search_graph(graph_id, query, limit=limit, scope="edges")
        logger.info(f"QuickSearch complete: {result.total_count} results")
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        """Interview OASIS simulation agents. This is OASIS-specific, not graph-related."""
        from .simulation_runner import SimulationRunner

        logger.info(f"InterviewAgents: {interview_requirement[:50]}")

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        profiles = self._load_agent_profiles(simulation_id)
        if not profiles:
            result.summary = "No agent profiles found"
            return result

        result.total_agents = len(profiles)

        selected_agents, selected_indices, reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents,
        )
        result.selected_agents = selected_agents
        result.selection_reasoning = reasoning

        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents,
            )

        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])

        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Drawing on your persona, memories, and actions, "
            "answer the following questions in plain text.\n"
            "Requirements:\n"
            "1. Answer directly, no tool calls\n"
            "2. No JSON format\n"
            "3. No markdown headers\n"
            "4. Number each answer\n"
            "5. Separate answers with blank lines\n"
            "6. At least 2-3 sentences per answer\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt,
                })

            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0,
            )

            if not api_result.get("success", False):
                result.summary = f"Interview API failed: {api_result.get('error', 'unknown')}"
                return result

            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")

                twitter_response = results_dict.get(f"twitter_{agent_idx}", {}).get("response", "")
                reddit_response = results_dict.get(f"reddit_{agent_idx}", {}).get("response", "")

                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                response_text = (
                    f"[Twitter]\n{twitter_response or '(no response)'}\n\n"
                    f"[Reddit]\n{reddit_response or '(no response)'}"
                )

                import re
                combined_responses = f"{twitter_response} {reddit_response}"
                sentences = re.split(r'[.!?]', combined_responses)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                ]
                key_quotes = [s + "." for s in meaningful[:3]]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5],
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            result.summary = f"Interview failed: {e}. Simulation environment may be stopped."
            return result
        except Exception as e:
            logger.error(f"Interview exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Interview error: {e}"
            return result

        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement,
            )
        return result

    # --- Private helper methods ---

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5,
    ) -> List[str]:
        system_prompt = (
            "You are a research analyst. Decompose the query into sub-questions "
            "for comprehensive knowledge graph search. Return JSON: {\"queries\": [...]}"
        )
        user_prompt = f"Query: {query}\nContext: {simulation_requirement}\n{report_context}"
        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            return response.get("queries", [query])[:max_queries]
        except Exception as e:
            logger.warning(f"Sub-query generation failed: {e}")
            return [query]

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        import os
        import csv

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f"../../uploads/simulations/{simulation_id}",
        )
        profiles = []

        reddit_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_path):
            try:
                with open(reddit_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        twitter_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_path):
            try:
                with open(twitter_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unknown",
                        })
            except Exception:
                pass

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ) -> tuple:
        agent_summaries = []
        for i, p in enumerate(profiles):
            agent_summaries.append({
                "index": i,
                "name": p.get("realname", p.get("username", f"Agent_{i}")),
                "profession": p.get("profession", "Unknown"),
                "bio": p.get("bio", "")[:200],
            })

        system_prompt = (
            "Select the most relevant agents for the interview. "
            "Return JSON: {\"selected_indices\": [...], \"reasoning\": \"...\"}"
        )
        user_prompt = (
            f"Interview: {interview_requirement}\n"
            f"Context: {simulation_requirement}\n"
            f"Agents: {json.dumps(agent_summaries)}\n"
            f"Select up to {max_agents}."
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Auto-selected")
            selected = []
            valid_indices = []
            for idx in indices:
                if 0 <= idx < len(profiles):
                    selected.append(profiles[idx])
                    valid_indices.append(idx)
            return selected, valid_indices, reasoning
        except Exception:
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Default selection"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        roles = [a.get("profession", "Unknown") for a in selected_agents]
        system_prompt = (
            "Generate 3-5 interview questions. Open-ended, multi-perspective. "
            "Return JSON: {\"questions\": [...]}"
        )
        user_prompt = f"Topic: {interview_requirement}\nContext: {simulation_requirement}\nRoles: {', '.join(roles)}"
        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )
            return response.get("questions", [f"What is your view on {interview_requirement}?"])
        except Exception:
            return [
                f"What is your perspective on {interview_requirement}?",
                "How does this affect you or your group?",
                "What do you think should be done about this?",
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str,
    ) -> str:
        if not interviews:
            return "No interviews completed"

        interview_texts = []
        for iv in interviews:
            interview_texts.append(f"[{iv.agent_name} ({iv.agent_role})]\n{iv.response[:500]}")

        system_prompt = (
            "Summarize the interview findings. Highlight consensus, divergences, "
            "and notable quotes. Keep under 1000 words. Plain text, no markdown headers."
        )
        user_prompt = f"Topic: {interview_requirement}\n\nInterviews:\n{''.join(interview_texts)}"

        try:
            return self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
        except Exception:
            return f"Interviewed {len(interviews)} agents: " + ", ".join([i.agent_name for i in interviews])

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        if not response or not response.strip().startswith("{"):
            return response
        text = response.strip()
        if "tool_name" not in text[:80]:
            return response
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "arguments" in data:
                for key in ("content", "text", "body", "message", "reply"):
                    if key in data["arguments"]:
                        return str(data["arguments"][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            import re
            match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace("\\n", "\n").replace('\\"', '"')
        return response

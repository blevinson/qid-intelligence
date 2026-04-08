"""
Financial Intelligence API - Query enriched regime signals from Graphiti.
Exposes endpoints for the bot fleet to consume sentiment-enriched regime data.
"""

import asyncio
from flask import request, jsonify
from . import intelligence_bp
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('qid.intelligence_api')


def _get_graphiti():
    """Lazy-init Graphiti client for search queries."""
    neo4j_uri = Config.NEO4J_URI
    if not neo4j_uri:
        return None

    from graphiti_core import Graphiti
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
    return Graphiti(
        neo4j_uri,
        Config.NEO4J_USER,
        Config.NEO4J_PASSWORD,
        llm_client=OpenAIClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config),
    )


@intelligence_bp.route('/regime/search', methods=['POST'])
def regime_search():
    """
    Search the knowledge graph for regime-related intelligence.

    POST body:
        { "query": "What is the current market regime?", "limit": 10 }

    Returns structured facts from Graphiti.
    """
    data = request.get_json() or {}
    query = data.get("query", "current market regime")
    limit = data.get("limit", 10)
    group_id = data.get("group_id", "qid_intelligence")

    graphiti = _get_graphiti()
    if not graphiti:
        return jsonify({"error": "Graphiti not configured (NEO4J_URI missing)"}), 503

    try:
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(
            graphiti.search(query, num_results=limit, group_ids=[group_id])
        )
        loop.close()

        facts = []
        for edge in results:
            facts.append({
                "fact": edge.fact if hasattr(edge, "fact") else str(edge),
                "source": edge.episodes[0] if hasattr(edge, "episodes") and edge.episodes else None,
                "created_at": edge.created_at.isoformat() if hasattr(edge, "created_at") and edge.created_at else None,
            })

        return jsonify({"query": query, "facts": facts, "count": len(facts)})
    except Exception as e:
        logger.error(f"Regime search error: {e}")
        return jsonify({"error": str(e)}), 500


@intelligence_bp.route('/regime/entities', methods=['GET'])
def regime_entities():
    """
    Get all entities in the knowledge graph.
    Returns entity names and types.
    """
    if not Config.NEO4J_URI:
        return jsonify({"error": "NEO4J_URI not configured"}), 503

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) RETURN n.name AS name, labels(n) AS labels, "
                "n.summary AS summary ORDER BY n.name LIMIT 100"
            )
            entities = [
                {
                    "name": r["name"],
                    "labels": r["labels"],
                    "summary": r["summary"],
                }
                for r in result
            ]
        return jsonify({"entities": entities, "count": len(entities)})
    except Exception as e:
        logger.error(f"Entity query error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        driver.close()


@intelligence_bp.route('/regime/sentiment', methods=['GET'])
def regime_sentiment():
    """
    Get aggregated sentiment from financial simulation episodes.
    Queries the graph for recent agent commentary and extracts
    bullish/bearish signals.
    """
    graphiti = _get_graphiti()
    if not graphiti:
        return jsonify({"error": "Graphiti not configured"}), 503

    try:
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(
            graphiti.search(
                "financial simulation agent market sentiment bullish bearish",
                num_results=20,
                group_ids=["qid_intelligence"],
            )
        )
        loop.close()

        facts = []
        for edge in results:
            fact_text = edge.fact if hasattr(edge, "fact") else str(edge)
            facts.append(fact_text)

        bullish_keywords = {"bullish", "rally", "support", "accumulation", "upside", "buy"}
        bearish_keywords = {"bearish", "selloff", "resistance", "distribution", "downside", "sell"}

        bullish_count = sum(
            1 for f in facts
            if any(k in f.lower() for k in bullish_keywords)
        )
        bearish_count = sum(
            1 for f in facts
            if any(k in f.lower() for k in bearish_keywords)
        )
        total = bullish_count + bearish_count or 1

        return jsonify({
            "bullish_ratio": round(bullish_count / total, 2),
            "bearish_ratio": round(bearish_count / total, 2),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "total_facts": len(facts),
            "sample_facts": facts[:5],
        })
    except Exception as e:
        logger.error(f"Sentiment query error: {e}")
        return jsonify({"error": str(e)}), 500

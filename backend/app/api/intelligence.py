"""
Financial Intelligence API - Query enriched regime signals from Graphiti.
Exposes endpoints for the bot fleet to consume sentiment-enriched regime data.
"""

import asyncio
import math
from datetime import datetime, timezone
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


def _get_tsdb_connection():
    """Get a psycopg2 connection to TimescaleDB."""
    if not Config.TSDB_HOST:
        return None
    import psycopg2
    return psycopg2.connect(
        host=Config.TSDB_HOST,
        port=Config.TSDB_PORT,
        dbname=Config.TSDB_DATABASE,
        user=Config.TSDB_USER,
        password=Config.TSDB_PASSWORD,
        connect_timeout=5,
    )


def _time_decay_weight(created_at, now, half_life_hours=6.0):
    """Exponential decay: weight = exp(-lambda * age_hours). Half-life default = 6h."""
    if created_at is None:
        return 0.0
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_seconds = max((now - created_at).total_seconds(), 0)
    age_hours = age_seconds / 3600.0
    decay_lambda = math.log(2) / half_life_hours
    return math.exp(-decay_lambda * age_hours)


@intelligence_bp.route('/regime/current', methods=['GET'])
def regime_current():
    """
    Time-decayed regime consensus for bot fleet consumption.

    Returns:
        - tsdb: latest macro_sweep from TimescaleDB (authoritative regime source)
        - graph_facts: time-weighted facts from Graphiti (regime/VIX/sentiment related)
        - persona_sentiment: aggregated bullish/bearish from recent simulation episodes
        - consensus: combined signal with confidence score
    """
    now = datetime.now(timezone.utc)
    half_life = float(request.args.get('half_life_hours', 6.0))
    result = {
        "timestamp": now.isoformat(),
        "tsdb": None,
        "graph_facts": [],
        "persona_sentiment": {"bullish_ratio": 0.5, "bearish_ratio": 0.5, "total_facts": 0},
        "consensus": {"regime": "unknown", "confidence": 0.0, "bias": "neutral"},
    }

    # 1) Authoritative regime from TSDB macro_sweeps
    tsdb_regime = None
    try:
        conn = _get_tsdb_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT time, regime, vix, sp500_change_pct, wti,
                       bias_direction, suppress, threshold
                FROM macro_sweeps ORDER BY time DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                tsdb_regime = {
                    "time": row[0].isoformat() if row[0] else None,
                    "regime": row[1],
                    "vix": float(row[2]) if row[2] is not None else None,
                    "sp500_change_pct": float(row[3]) if row[3] is not None else None,
                    "wti": float(row[4]) if row[4] is not None else None,
                    "bias_direction": row[5],
                    "suppress": row[6],
                    "threshold": float(row[7]) if row[7] is not None else None,
                }
                result["tsdb"] = tsdb_regime
            conn.close()
    except Exception as e:
        logger.warning(f"TSDB query failed: {e}")

    # 2) Time-decayed facts from Neo4j (direct Cypher, no LLM call)
    try:
        if Config.NEO4J_URI:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            )
            with driver.session() as session:
                # Get regime-related facts from the last 24h
                cypher = """
                MATCH (s)-[r:RELATES_TO]->(t)
                WHERE r.created_at > datetime() - duration({hours: 24})
                  AND (r.fact CONTAINS 'regime' OR r.fact CONTAINS 'VIX'
                       OR r.fact CONTAINS 'bullish' OR r.fact CONTAINS 'bearish'
                       OR r.fact CONTAINS 'market' OR r.fact CONTAINS 'sentiment')
                RETURN r.fact AS fact, r.created_at AS created_at,
                       s.name AS source, t.name AS target
                ORDER BY r.created_at DESC
                LIMIT 30
                """
                records = list(session.run(cypher))

            driver.close()

            weighted_facts = []
            for rec in records:
                created = rec["created_at"]
                if hasattr(created, 'to_native'):
                    created = created.to_native()
                weight = _time_decay_weight(created, now, half_life)
                weighted_facts.append({
                    "fact": rec["fact"],
                    "source": rec["source"],
                    "target": rec["target"],
                    "created_at": created.isoformat() if created else None,
                    "weight": round(weight, 4),
                })
            result["graph_facts"] = weighted_facts
    except Exception as e:
        logger.warning(f"Neo4j query failed: {e}")

    # 3) Persona sentiment from recent simulation episodes
    try:
        if Config.NEO4J_URI:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
            )
            with driver.session() as session:
                cypher = """
                MATCH (e:Episodic)
                WHERE e.created_at > datetime() - duration({hours: 24})
                  AND e.source_description CONTAINS 'simulation'
                RETURN e.content AS content, e.created_at AS created_at
                ORDER BY e.created_at DESC
                LIMIT 10
                """
                sim_records = list(session.run(cypher))
            driver.close()

            bullish_kw = {"bullish", "rally", "support", "accumulation", "upside", "buy", "bid"}
            bearish_kw = {"bearish", "selloff", "resistance", "distribution", "downside", "sell", "offer"}

            weighted_bull = 0.0
            weighted_bear = 0.0
            total_weight = 0.0

            for rec in sim_records:
                content = (rec["content"] or "").lower()
                created = rec["created_at"]
                if hasattr(created, 'to_native'):
                    created = created.to_native()
                w = _time_decay_weight(created, now, half_life)

                bull = sum(1 for k in bullish_kw if k in content)
                bear = sum(1 for k in bearish_kw if k in content)
                if bull + bear > 0:
                    weighted_bull += w * bull
                    weighted_bear += w * bear
                    total_weight += w

            if total_weight > 0:
                bull_ratio = round(weighted_bull / (weighted_bull + weighted_bear), 3)
                bear_ratio = round(1 - bull_ratio, 3)
            else:
                bull_ratio = 0.5
                bear_ratio = 0.5

            result["persona_sentiment"] = {
                "bullish_ratio": bull_ratio,
                "bearish_ratio": bear_ratio,
                "total_episodes": len(sim_records),
                "total_weight": round(total_weight, 3),
            }
    except Exception as e:
        logger.warning(f"Sentiment query failed: {e}")

    # 4) Build consensus
    regime = tsdb_regime["regime"] if tsdb_regime else "unknown"
    confidence = 0.0

    # TSDB is authoritative -- if we have it, high base confidence
    if tsdb_regime:
        confidence = 0.7
        # Boost confidence if graph facts agree
        regime_facts = [f for f in result["graph_facts"] if "regime" in f["fact"].lower()]
        agreeing = sum(1 for f in regime_facts if regime in f["fact"].lower())
        if regime_facts:
            confidence += 0.15 * (agreeing / len(regime_facts))

        # Slight boost/penalty from sentiment alignment
        bull_ratio = result["persona_sentiment"]["bullish_ratio"]
        if regime == "elevated" and bull_ratio < 0.4:
            confidence += 0.05  # personas see risk in elevated regime
        elif regime == "normal" and bull_ratio > 0.6:
            confidence += 0.05  # personas confirm calm conditions

    # Bias from persona sentiment
    bull_ratio = result["persona_sentiment"]["bullish_ratio"]
    if bull_ratio > 0.65:
        bias = "bullish"
    elif bull_ratio < 0.35:
        bias = "bearish"
    else:
        bias = "neutral"

    result["consensus"] = {
        "regime": regime,
        "confidence": round(min(confidence, 1.0), 3),
        "bias": bias,
        "suppress": tsdb_regime.get("suppress", False) if tsdb_regime else False,
    }

    return jsonify(result)


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

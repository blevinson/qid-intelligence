#!/usr/bin/env python3
"""Bulk-load Alpaca tradable universe into Graphiti's neo4j as the
structural ticker / industry / sector backbone.

QIDP-226 follow-up. Pure Cypher MERGEs, no embeddings, no LLM. Idempotent.

Schema written:
    (:Ticker {symbol, name, exchange, market_cap, group_id})
    (:Industry {name, group_id})
    (:Sector {name, group_id})
    (:Ticker)-[:IN_INDUSTRY]->(:Industry)
    (:Industry)-[:IN_SECTOR]->(:Sector)
    (:Ticker)-[:IN_SECTOR]->(:Sector)         # kept for compat with bridge

group_id = 'qid_intelligence' so existing crucix idea episodes can traverse
to these nodes via name match (Bridge MERGEs Ticker by {symbol, group_id}).

Source: alpaca_x_industry view in qid_analytics.

Env:
    QID_TSDB_DSN | QID_DB_*
    NEO4J_URI    (default bolt://neo4j.qid.svc.cluster.local:7687)
    NEO4J_USER   (default neo4j)
    NEO4J_PASSWORD
    GRAPHITI_GROUP_ID (default qid_intelligence)
"""

from __future__ import annotations

import logging
import os
import sys

import psycopg2
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("graphiti_universe_load")

DSN = os.environ.get("QID_TSDB_DSN") or (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid_analytics')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', '')}"
)
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j.qid.svc.cluster.local:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "").lstrip("neo4j/")
GROUP_ID = os.environ.get("GRAPHITI_GROUP_ID", "qid_intelligence")

BATCH_SIZE = 500


def fetch_universe() -> list[dict]:
    sql = """
        SELECT
            x.alpaca_symbol AS symbol,
            x.name,
            x.exchange_alpaca AS exchange,
            x.sector,
            x.industry,
            x.is_etf,
            m.market_cap
        FROM alpaca_x_industry x
        LEFT JOIN ticker_metadata m
          ON m.symbol = COALESCE(x.fmp_symbol, x.alpaca_symbol)
        WHERE x.tradable
          AND x.sector IS NOT NULL
          AND x.industry IS NOT NULL
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    log.info("fetched %d ticker→industry→sector rows from TSDB", len(rows))
    return rows


CYPHER_BATCH = """
UNWIND $rows AS r
MERGE (s:Sector {name: r.sector, group_id: $gid})
MERGE (i:Industry {name: r.industry, group_id: $gid})
MERGE (i)-[:IN_SECTOR]->(s)
MERGE (t:Ticker {symbol: r.symbol, group_id: $gid})
  ON CREATE SET t.name = r.name,
                t.exchange = r.exchange,
                t.market_cap = r.market_cap,
                t.is_etf = r.is_etf,
                t.created_at = datetime()
  ON MATCH  SET t.name = COALESCE(r.name, t.name),
                t.exchange = COALESCE(r.exchange, t.exchange),
                t.market_cap = COALESCE(r.market_cap, t.market_cap),
                t.is_etf = COALESCE(r.is_etf, t.is_etf),
                t.updated_at = datetime()
MERGE (t)-[:IN_INDUSTRY]->(i)
MERGE (t)-[:IN_SECTOR]->(s)
"""


def load(rows: list[dict]) -> None:
    if not rows:
        log.warning("no rows to load")
        return
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with drv.session() as sess:
            for i in range(0, len(rows), BATCH_SIZE):
                chunk = rows[i:i + BATCH_SIZE]
                sess.run(CYPHER_BATCH, rows=chunk, gid=GROUP_ID)
                log.info("merged batch %d-%d / %d", i, i + len(chunk), len(rows))
    finally:
        drv.close()


def report() -> None:
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    q = """
        MATCH (t:Ticker {group_id: $gid}) WITH count(t) AS tickers
        MATCH (i:Industry {group_id: $gid}) WITH tickers, count(i) AS industries
        MATCH (s:Sector {group_id: $gid}) WITH tickers, industries, count(s) AS sectors
        MATCH (:Ticker {group_id: $gid})-[r1:IN_INDUSTRY]->()
        WITH tickers, industries, sectors, count(r1) AS in_industry
        MATCH (:Industry {group_id: $gid})-[r2:IN_SECTOR]->()
        WITH tickers, industries, sectors, in_industry, count(r2) AS industry_in_sector
        MATCH (:Ticker {group_id: $gid})-[r3:IN_SECTOR]->()
        RETURN tickers, industries, sectors, in_industry, industry_in_sector,
               count(r3) AS ticker_in_sector
    """
    try:
        with drv.session() as sess:
            rec = sess.run(q, gid=GROUP_ID).single()
            if rec:
                log.info("graph state (group=%s): %s", GROUP_ID, dict(rec))
    finally:
        drv.close()


def main() -> int:
    if not NEO4J_PASSWORD:
        log.error("NEO4J_PASSWORD required")
        return 2
    rows = fetch_universe()
    load(rows)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())

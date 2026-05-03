#!/usr/bin/env python3
"""FIN-1155: Archive crucix trade ideas from graphiti to TimescaleDB.

Reads crucix_idea_* episodes from the qid_intelligence graphiti group and
inserts any that are missing from the crucix_trade_ideas TSDB table.  Runs
as a CronJob every 15 minutes so the TSDB stays in sync with graphiti even
before the bridge image is rebuilt with native inline archival.

Usage:
    python crucix_trade_ideas_archive.py [--max-episodes N] [--dry-run]

Env (same as crucix_idea_outcome.py):
    QID_DB_HOST, QID_DB_PORT, QID_DB_NAME, QID_DB_USER, QID_DB_PASS
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    GRAPHITI_GROUP_ID  (default: qid_intelligence)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig

# The bridge image ships a custom LLM client at /app/llm/. Add the app dir
# to the path so we can import it without modifying the image.
_BRIDGE_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
if os.path.isdir(os.path.join(_BRIDGE_APP_DIR, "llm")):
    sys.path.insert(0, os.path.abspath(_BRIDGE_APP_DIR))
elif os.path.isdir("/app/llm"):
    sys.path.insert(0, "/app")

try:
    from llm import ClaudeCodeLLMClient, NoopCrossEncoder, TEIEmbedder
    _CUSTOM_LLM = True
except ImportError:
    _CUSTOM_LLM = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TSDB_DSN = (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid_analytics')} "
    f"user={os.environ.get('QID_DB_USER', '')} "
    f"password={os.environ.get('QID_DB_PASS', '')}"
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j.qid.svc.cluster.local:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
_raw_neo4j_pw = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_PASSWORD = _raw_neo4j_pw.split("/", 1)[1] if "/" in _raw_neo4j_pw else _raw_neo4j_pw
GRAPHITI_GROUP_ID = os.environ.get("GRAPHITI_GROUP_ID", "qid_intelligence")

log = logging.getLogger("crucix_trade_ideas_archive")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# ---------------------------------------------------------------------------
# TSDB helpers
# ---------------------------------------------------------------------------

ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS crucix_trade_ideas (
    time        TIMESTAMPTZ NOT NULL,
    idea_name   TEXT        NOT NULL,
    ticker      TEXT        NOT NULL,
    direction   TEXT,
    confidence  TEXT,
    title       TEXT,
    thesis      TEXT,
    risk        TEXT,
    horizon     TEXT,
    signals     JSONB,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (idea_name, time)
);
"""

INSERT_SQL = """
INSERT INTO crucix_trade_ideas
    (time, idea_name, ticker, direction, confidence,
     title, thesis, risk, horizon, signals)
VALUES
    (%(time)s, %(idea_name)s, %(ticker)s, %(direction)s, %(confidence)s,
     %(title)s, %(thesis)s, %(risk)s, %(horizon)s, %(signals)s)
ON CONFLICT (idea_name, time) DO NOTHING;
"""


def _tsdb_connect():
    return psycopg2.connect(TSDB_DSN)


def _ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(ENSURE_TABLE_SQL)
        # Create hypertable if TimescaleDB is available; graceful fallback.
        try:
            cur.execute(
                "SELECT create_hypertable('crucix_trade_ideas', 'time', "
                "if_not_exists => TRUE, migrate_data => TRUE)"
            )
        except Exception:
            conn.rollback()
    conn.commit()


def _load_archived_names(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT idea_name FROM crucix_trade_ideas")
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_IDEA_NAME_RE = re.compile(r"^crucix_idea_([A-Z0-9]+)_(\d{8})_(\d{6})$")
_DIRECTION_RE = re.compile(
    r"crucix trade idea:\s*(LONG|SHORT|HEDGE|WATCH|AVOID)\s+(\S+?)(?:\s+\([^)]+\))?,\s*confidence\s+(\w+)",
    re.IGNORECASE,
)
_THESIS_RE = re.compile(r"Thesis:\s*(.+?)(?:\s+Catalysts:|Risks:|Levels:|Sources:|$)", re.IGNORECASE | re.DOTALL)
_RISK_RE = re.compile(r"Risks:\s*(.+?)(?:\s+Catalysts:|Levels:|Sources:|$)", re.IGNORECASE | re.DOTALL)


def _parse_idea_name(name: str) -> tuple[str, datetime] | None:
    m = _IDEA_NAME_RE.match(name)
    if not m:
        return None
    ticker, date_s, time_s = m.group(1), m.group(2), m.group(3)
    try:
        idea_dt = datetime.strptime(f"{date_s}{time_s}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return ticker, idea_dt


def _parse_episode(episode: dict) -> dict[str, Any] | None:
    name = episode.get("name", "") or ""
    content = episode.get("content", "") or ""

    parsed_name = _parse_idea_name(name)
    if not parsed_name:
        return None
    ticker, idea_dt = parsed_name

    m = _DIRECTION_RE.search(content)
    direction = m.group(1).upper() if m else None
    confidence = m.group(3).upper() if m else None

    thesis_m = _THESIS_RE.search(content)
    thesis = thesis_m.group(1).strip() if thesis_m else None

    risk_m = _RISK_RE.search(content)
    risk = risk_m.group(1).strip() if risk_m else None

    return {
        "time": idea_dt,
        "idea_name": name,
        "ticker": ticker,
        "direction": direction,
        "confidence": confidence,
        "title": None,
        "thesis": thesis,
        "risk": risk,
        "horizon": None,
        "signals": None,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(max_episodes: int, dry_run: bool) -> None:
    log.info("crucix_trade_ideas_archive starting", extra={"max_episodes": max_episodes, "dry_run": dry_run})

    LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")
    if _CUSTOM_LLM:
        llm_cfg = LLMConfig(model=LLM_MODEL, small_model=LLM_MODEL)
        graphiti = Graphiti(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
            llm_client=ClaudeCodeLLMClient(config=llm_cfg),
            embedder=TEIEmbedder(),
            cross_encoder=NoopCrossEncoder(),
        )
    else:
        graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    conn = _tsdb_connect()
    _ensure_table(conn)
    already_archived = _load_archived_names(conn)
    log.info("already archived", extra={"count": len(already_archived)})

    # Query neo4j directly for crucix_idea_* episodes in the group.
    # graphiti v0.29.0 has no get_episodes_by_group_id; use driver Cypher.
    cypher = """
        MATCH (e:Episodic {group_id: $gid})
        WHERE e.name STARTS WITH 'crucix_idea_'
          AND NOT e.name STARTS WITH 'crucix_idea_outcome_'
        RETURN e.name AS name, e.content AS content
        LIMIT $limit
    """
    idea_episodes: list[dict] = []
    async with graphiti.driver.session() as sess:
        result = await sess.run(cypher, gid=GRAPHITI_GROUP_ID, limit=max_episodes)
        async for record in result:
            idea_episodes.append({
                "name": record["name"],
                "content": record["content"] or "",
            })
    log.info("total idea episodes in graphiti", extra={"count": len(idea_episodes)})

    pending = [e for e in idea_episodes if e["name"] not in already_archived]
    log.info("pending archive", extra={"count": len(pending)})

    inserted = skipped = errors = 0
    for ep in pending:
        row = _parse_episode(ep)
        if row is None:
            skipped += 1
            continue
        if dry_run:
            log.info("dry-run: would insert", extra={"idea_name": row["idea_name"]})
            inserted += 1
            continue
        try:
            with conn.cursor() as cur:
                cur.execute(INSERT_SQL, row)
            conn.commit()
            inserted += 1
        except Exception as exc:
            conn.rollback()
            log.error("insert failed", extra={"idea_name": row["idea_name"], "error": str(exc)})
            errors += 1

    conn.close()
    await graphiti.close()
    log.info(
        "crucix_trade_ideas_archive done",
        extra={"inserted": inserted, "skipped": skipped, "errors": errors},
    )
    if errors:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive crucix trade ideas from graphiti to TSDB (FIN-1155)")
    parser.add_argument("--max-episodes", type=int, default=5000,
                        help="Max episodes to fetch from graphiti (default: 5000)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and log; don't insert")
    args = parser.parse_args()

    asyncio.run(run(args.max_episodes, args.dry_run))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""QIDP-241: Crucix idea outcome attribution — forward-return write-back.

Hourly cronjob that:
1. Queries graphiti (qid_intelligence) for crucix_idea episodes older than
   horizon_days (default 5d) with no outcome yet (idempotent via TSDB dedup).
2. Parses (ticker, direction, confidence) from episode content.
3. Loads OHLCV bars from MinIO for the horizon window.
4. Computes: realized_return_pct, hit (bool), mfe_pct, mae_pct, bars_to_outcome.
5. Tags direction_mismatch when thesis polarity contradicts direction.
6. Writes IDEA_OUTCOME episode to graphiti (qid_intelligence group).
7. Inserts into TSDB crucix_idea_outcomes for idempotency + analytics.

Usage:
    python crucix_idea_outcome.py [--horizon-days N] [--backfill-date YYYY-MM-DD]

Env (inherits panel_build conventions):
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE
    QID_EQUITY_BUCKET
    QID_DB_HOST, QID_DB_PORT, QID_DB_NAME, QID_DB_USER, QID_DB_PASS
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    GRAPHITI_GROUP_ID  (default: qid_intelligence)
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from minio import Minio
from minio.error import S3Error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio.qid.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
QID_EQUITY_BUCKET = os.environ.get("QID_EQUITY_BUCKET", "qid-equity")

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

DEFAULT_HORIZON_DAYS = 5

# Negative thesis words that contradict a LONG signal (or positive for SHORT).
_BEARISH_RE = re.compile(
    r"\b(unfriendly|headwind|drag|poor|lagging|lag|not supportive|weakening|weak|"
    r"negative|bearish|pressure|risk looks|downside|below|underperform|"
    r"unfavorable|deteriorat)\b",
    re.IGNORECASE,
)
_BULLISH_RE = re.compile(
    r"\b(surge|bullish|supportive|strong|breakout|momentum|outperform|"
    r"upside|positive|growth|favorable|above|accelerat|lift|rally)\b",
    re.IGNORECASE,
)

log = logging.getLogger("crucix_idea_outcome")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_IDEA_NAME_RE = re.compile(r"^crucix_idea_([A-Z0-9]+)_(\d{8})_(\d{6})$")
_DIRECTION_RE = re.compile(r"crucix trade idea:\s*(LONG|SHORT)\s+(\S+),\s*confidence\s+(\w+)", re.IGNORECASE)


def parse_idea_name(name: str) -> tuple[str, datetime] | None:
    """Extract (ticker, idea_dt) from a crucix_idea episode name."""
    m = _IDEA_NAME_RE.match(name)
    if not m:
        return None
    ticker, date_s, time_s = m.group(1), m.group(2), m.group(3)
    try:
        idea_dt = datetime.strptime(f"{date_s}{time_s}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return ticker, idea_dt


def parse_idea_content(content: str) -> dict[str, str] | None:
    """Extract direction, ticker, confidence from content string."""
    m = _DIRECTION_RE.search(content)
    if not m:
        return None
    return {
        "direction": m.group(1).upper(),
        "ticker": m.group(2).upper().rstrip(".,"),
        "confidence": m.group(3).upper(),
    }


def detect_direction_mismatch(direction: str, content: str) -> bool:
    """Return True when thesis polarity contradicts the direction tag.

    Counts bearish/bullish signal words in the thesis portion of the content.
    LONG + majority bearish = mismatch; SHORT + majority bullish = mismatch.
    """
    # Extract thesis (everything after the first sentence)
    thesis_start = content.find("Thesis:")
    thesis = content[thesis_start:] if thesis_start != -1 else content
    bearish = len(_BEARISH_RE.findall(thesis))
    bullish = len(_BULLISH_RE.findall(thesis))
    if direction == "LONG" and bearish > bullish:
        return True
    if direction == "SHORT" and bullish > bearish:
        return True
    return False


# ---------------------------------------------------------------------------
# MinIO / OHLCV
# ---------------------------------------------------------------------------

def _minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def load_bars(minio: Minio, ticker: str) -> pd.DataFrame | None:
    """Load daily OHLCV bars for ticker from MinIO. Returns None if absent."""
    key = f"bars/{ticker}.parquet"
    try:
        resp = minio.get_object(QID_EQUITY_BUCKET, key)
        try:
            buf = io.BytesIO(resp.read())
        finally:
            resp.close()
            resp.release_conn()
        df = pd.read_parquet(buf)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date").reset_index(drop=True)
    except S3Error as e:
        if e.code == "NoSuchKey":
            return None
        raise


# ---------------------------------------------------------------------------
# Outcome computation
# ---------------------------------------------------------------------------

def compute_outcome(
    bars: pd.DataFrame,
    idea_date: date,
    direction: str,
    horizon_days: int,
) -> dict[str, Any] | None:
    """Compute return metrics over the horizon window starting the day after idea_date.

    Returns None when bars data is insufficient.
    """
    # Entry: closing price on idea_date (or nearest prior bar)
    prior = bars[bars["date"] <= idea_date]
    if prior.empty:
        return None
    entry_row = prior.iloc[-1]
    entry_price = float(entry_row.get("adjusted_close", entry_row.get("close", 0)))
    if entry_price <= 0:
        return None

    # Horizon window: next horizon_days trading bars after the idea date
    future = bars[bars["date"] > idea_date].head(horizon_days)
    if future.empty:
        return None

    exit_row = future.iloc[-1]
    exit_price = float(exit_row.get("adjusted_close", exit_row.get("close", entry_price)))

    realized_return = (exit_price - entry_price) / entry_price
    if direction == "SHORT":
        realized_return = -realized_return

    # MFE / MAE — use daily high/low when available
    highs = future.get("high", future.get("adjusted_close", future.iloc[:, 0])).astype(float)
    lows = future.get("low", future.get("adjusted_close", future.iloc[:, 0])).astype(float)

    if direction == "LONG":
        mfe = (highs.max() - entry_price) / entry_price
        mae = (lows.min() - entry_price) / entry_price
    else:  # SHORT
        mfe = (entry_price - lows.min()) / entry_price
        mae = (entry_price - highs.max()) / entry_price

    # bars_to_outcome: first bar where realized gain > 0 (hits profitable)
    bars_to_outcome = len(future)
    for i, (_, row) in enumerate(future.iterrows()):
        close = float(row.get("adjusted_close", row.get("close", entry_price)))
        bar_return = (close - entry_price) / entry_price
        if direction == "SHORT":
            bar_return = -bar_return
        if bar_return > 0:
            bars_to_outcome = i + 1
            break

    return {
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "realized_return_pct": round(realized_return * 100, 4),
        "hit": realized_return > 0,
        "mfe_pct": round(mfe * 100, 4),
        "mae_pct": round(mae * 100, 4),
        "bars_to_outcome": bars_to_outcome,
        "horizon_days": horizon_days,
        "entry_date": str(idea_date),
        "exit_date": str(exit_row["date"]),
    }


# ---------------------------------------------------------------------------
# TSDB — idempotency table
# ---------------------------------------------------------------------------

ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS crucix_idea_outcomes (
    idea_name           TEXT        NOT NULL,
    ticker              TEXT        NOT NULL,
    direction           TEXT        NOT NULL,
    confidence          TEXT,
    idea_dt             TIMESTAMPTZ NOT NULL,
    entry_date          DATE        NOT NULL,
    exit_date           DATE,
    entry_price         DOUBLE PRECISION,
    exit_price          DOUBLE PRECISION,
    realized_return_pct DOUBLE PRECISION,
    hit                 BOOLEAN,
    mfe_pct             DOUBLE PRECISION,
    mae_pct             DOUBLE PRECISION,
    bars_to_outcome     INTEGER,
    horizon_days        INTEGER,
    direction_mismatch  BOOLEAN     DEFAULT FALSE,
    processed_at        TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (idea_name)
);
"""

SELECT_PROCESSED = "SELECT idea_name FROM crucix_idea_outcomes"

INSERT_OUTCOME = """
INSERT INTO crucix_idea_outcomes (
    idea_name, ticker, direction, confidence, idea_dt,
    entry_date, exit_date, entry_price, exit_price,
    realized_return_pct, hit, mfe_pct, mae_pct, bars_to_outcome,
    horizon_days, direction_mismatch
) VALUES (
    %(idea_name)s, %(ticker)s, %(direction)s, %(confidence)s, %(idea_dt)s,
    %(entry_date)s, %(exit_date)s, %(entry_price)s, %(exit_price)s,
    %(realized_return_pct)s, %(hit)s, %(mfe_pct)s, %(mae_pct)s,
    %(bars_to_outcome)s, %(horizon_days)s, %(direction_mismatch)s
) ON CONFLICT (idea_name) DO NOTHING;
"""


def _tsdb_connect():
    return psycopg2.connect(TSDB_DSN)


def _ensure_table(cur) -> None:
    cur.execute(ENSURE_TABLE_SQL)


def _load_processed_set(cur) -> set[str]:
    cur.execute(SELECT_PROCESSED)
    return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Graphiti helpers
# ---------------------------------------------------------------------------

async def _get_recent_idea_episodes(
    graphiti: Graphiti,
    max_episodes: int = 2000,
) -> list[dict]:
    """Fetch crucix_idea_* episodes from qid_intelligence group."""
    episodes = await graphiti.get_episodes_by_group_id(
        group_id=GRAPHITI_GROUP_ID,
        max_episodes=max_episodes,
    )
    return [
        e for e in episodes
        if getattr(e, "name", "").startswith("crucix_idea_")
        # Exclude outcome episodes we previously wrote
        and not getattr(e, "name", "").startswith("crucix_idea_outcome_")
    ]


async def _write_outcome_episode(
    graphiti: Graphiti,
    idea_name: str,
    ticker: str,
    direction: str,
    confidence: str,
    outcome: dict[str, Any],
    direction_mismatch: bool,
    idea_dt: datetime,
) -> None:
    payload = {
        "idea_name": idea_name,
        "ticker": ticker,
        "direction": direction,
        "confidence": confidence,
        "realized_return_pct": outcome["realized_return_pct"],
        "hit": outcome["hit"],
        "mfe_pct": outcome["mfe_pct"],
        "mae_pct": outcome["mae_pct"],
        "bars_to_outcome": outcome["bars_to_outcome"],
        "horizon_days": outcome["horizon_days"],
        "entry_date": outcome["entry_date"],
        "exit_date": outcome["exit_date"],
        "direction_mismatch": direction_mismatch,
    }
    await graphiti.add_episode(
        name=f"crucix_idea_outcome_{ticker}_{idea_dt.strftime('%Y%m%d_%H%M%S')}",
        episode_body=json.dumps(payload),
        source=EpisodeType.json,
        source_description="Crucix idea outcome attribution (QIDP-241)",
        reference_time=datetime.now(timezone.utc),
        group_id=GRAPHITI_GROUP_ID,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(horizon_days: int, backfill_date: date | None) -> None:
    log.info("crucix_idea_outcome starting", extra={"horizon_days": horizon_days})

    graphiti = Graphiti(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    minio = _minio_client()

    with _tsdb_connect() as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            _ensure_table(cur)
            conn.commit()
            already_done = _load_processed_set(cur)

        log.info("already processed", extra={"count": len(already_done)})

        ideas = await _get_recent_idea_episodes(graphiti)
        cutoff = datetime.now(timezone.utc) - timedelta(days=horizon_days)

        pending = []
        for ep in ideas:
            name = ep.name
            if name in already_done:
                continue
            created_at = getattr(ep, "created_at", None)
            if created_at is None:
                continue
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if backfill_date is not None:
                # Backfill mode: process ideas created on or before backfill_date
                target_date = datetime.combine(backfill_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                if created_at > target_date + timedelta(days=1):
                    continue
            elif created_at > cutoff:
                # Standard mode: only ideas old enough for the horizon to have elapsed
                continue
            pending.append((ep, created_at))

        log.info("ideas pending attribution", extra={"count": len(pending)})
        processed = skipped = errors = 0

        bars_cache: dict[str, pd.DataFrame | None] = {}

        for ep, created_at in pending:
            name = ep.name
            content = getattr(ep, "content", "") or ""
            try:
                parsed_name = parse_idea_name(name)
                if not parsed_name:
                    skipped += 1
                    continue
                ticker, idea_dt = parsed_name

                parsed = parse_idea_content(content)
                if not parsed:
                    log.warning("could not parse content", extra={"name": name})
                    skipped += 1
                    continue

                direction = parsed["direction"]
                confidence = parsed["confidence"]

                # Load bars (cached per ticker)
                if ticker not in bars_cache:
                    bars_cache[ticker] = load_bars(minio, ticker)
                bars = bars_cache[ticker]
                if bars is None:
                    log.warning("no bars for ticker", extra={"ticker": ticker, "name": name})
                    skipped += 1
                    continue

                idea_date = idea_dt.date()
                outcome = compute_outcome(bars, idea_date, direction, horizon_days)
                if outcome is None:
                    log.warning("insufficient bars for outcome", extra={"ticker": ticker, "idea_date": str(idea_date)})
                    skipped += 1
                    continue

                direction_mismatch = detect_direction_mismatch(direction, content)

                # Write graphiti episode
                await _write_outcome_episode(
                    graphiti, name, ticker, direction, confidence,
                    outcome, direction_mismatch, idea_dt,
                )

                # Write TSDB dedup row
                with conn.cursor() as cur:
                    cur.execute(INSERT_OUTCOME, {
                        "idea_name": name,
                        "ticker": ticker,
                        "direction": direction,
                        "confidence": confidence,
                        "idea_dt": idea_dt,
                        "entry_date": date.fromisoformat(outcome["entry_date"]),
                        "exit_date": date.fromisoformat(outcome["exit_date"]),
                        "entry_price": outcome["entry_price"],
                        "exit_price": outcome["exit_price"],
                        "realized_return_pct": outcome["realized_return_pct"],
                        "hit": outcome["hit"],
                        "mfe_pct": outcome["mfe_pct"],
                        "mae_pct": outcome["mae_pct"],
                        "bars_to_outcome": outcome["bars_to_outcome"],
                        "horizon_days": outcome["horizon_days"],
                        "direction_mismatch": direction_mismatch,
                    })
                conn.commit()

                log.info(
                    "attributed",
                    extra={
                        "name": name,
                        "ticker": ticker,
                        "direction": direction,
                        "realized_return_pct": outcome["realized_return_pct"],
                        "hit": outcome["hit"],
                        "direction_mismatch": direction_mismatch,
                    },
                )
                processed += 1

            except Exception as exc:
                conn.rollback()
                log.error("attribution failed", extra={"name": name, "error": str(exc)})
                errors += 1

    await graphiti.close()
    log.info(
        "crucix_idea_outcome done",
        extra={"processed": processed, "skipped": skipped, "errors": errors},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crucix idea outcome attribution (QIDP-241)")
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument(
        "--backfill-date",
        type=date.fromisoformat,
        default=None,
        help="Process all ideas created on or before this date (e.g. 2026-05-02)",
    )
    args = parser.parse_args()

    asyncio.run(run(args.horizon_days, args.backfill_date))


if __name__ == "__main__":
    main()

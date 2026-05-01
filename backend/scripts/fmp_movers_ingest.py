#!/usr/bin/env python3
"""Hourly FMP movers snapshot → crucix_movers hypertable.

Hits three FMP endpoints (gainers, losers, most-actives), normalises, and
INSERTs a single time-stamped row per (time, category, symbol). Idempotent
across the same minute: PRIMARY KEY (time, category, symbol) blocks dups.

Env:
    FMP_API_KEY      — required
    QID_TSDB_DSN     — postgres connection string (or QID_DB_* parts)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

import httpx
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fmp_movers_ingest")

FMP_BASE = "https://financialmodelingprep.com/stable"
API_KEY = os.environ.get("FMP_API_KEY", "").strip()

DSN = os.environ.get("QID_TSDB_DSN") or (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid_analytics')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', '')}"
)

ENDPOINTS = [
    ("gainers",     "biggest-gainers"),
    ("losers",      "biggest-losers"),
    ("most_active", "most-actives"),
]


def fetch(category: str, ep: str) -> list[dict]:
    url = f"{FMP_BASE}/{ep}"
    log.info("fetching %s", url)
    with httpx.Client(timeout=30.0) as c:
        resp = c.get(url, params={"apikey": API_KEY})
    if resp.status_code != 200:
        raise RuntimeError(f"FMP HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    log.info("  %s rows=%d", category, len(data))
    return data


def normalize(category: str, data: list[dict], ts: datetime) -> list[dict]:
    rows = []
    for r in data:
        sym = (r.get("symbol") or "").strip()
        if not sym:
            continue
        rows.append({
            "time":       ts,
            "category":   category,
            "symbol":     sym,
            "name":       (r.get("name") or "").strip() or None,
            "price":      r.get("price"),
            "change_pct": r.get("changesPercentage"),
            "volume":     None,  # not in these endpoints; left NULL
            "exchange":   (r.get("exchange") or "").strip() or None,
        })
    return rows


def upsert(rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO crucix_movers (time, category, symbol, name, price, change_pct, volume, exchange)
        VALUES (%(time)s, %(category)s, %(symbol)s, %(name)s, %(price)s, %(change_pct)s, %(volume)s, %(exchange)s)
        ON CONFLICT (time, category, symbol) DO UPDATE SET
            name       = EXCLUDED.name,
            price      = EXCLUDED.price,
            change_pct = EXCLUDED.change_pct,
            volume     = COALESCE(EXCLUDED.volume, crucix_movers.volume),
            exchange   = EXCLUDED.exchange
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        execute_batch(cur, sql, rows, page_size=200)
    return len(rows)


def main() -> int:
    if not API_KEY:
        log.error("FMP_API_KEY required")
        return 2

    # Bucket all three categories to the same minute so one snapshot is one timestamp.
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    total = 0
    for category, ep in ENDPOINTS:
        try:
            data = fetch(category, ep)
        except Exception as e:
            log.warning("skip %s: %s", category, e)
            continue
        rows = normalize(category, data, ts)
        n = upsert(rows)
        total += n
        log.info("upserted category=%s rows=%d", category, n)

    # Brief intersection report against alpaca_x_industry.
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT m.category, COUNT(*) AS movers,
                   COUNT(DISTINCT i.industry) AS industries_touched
            FROM crucix_movers m
            LEFT JOIN alpaca_x_industry i ON i.alpaca_symbol = m.symbol
            WHERE m.time = %s
            GROUP BY 1 ORDER BY 1
        """, (ts,))
        for row in cur.fetchall():
            log.info("snapshot %s: %s movers, %s industries", row[0], row[1], row[2])
    log.info("total upserted=%d at ts=%s", total, ts.isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())

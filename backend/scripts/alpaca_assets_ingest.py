#!/usr/bin/env python3
"""Pull Alpaca's tradable asset list → upsert alpaca_assets.

Idempotent. Run as one-shot Job and daily via CronJob. The list churns
slowly (corporate actions, new listings) so daily is plenty.

Env:
    ALPACA_API_KEY       — required (paper key is fine)
    ALPACA_SECRET_KEY    — required
    ALPACA_BASE_URL      — optional, default https://paper-api.alpaca.markets
    QID_TSDB_DSN         — postgres connection string (or QID_DB_* parts)
"""

from __future__ import annotations

import logging
import os
import sys

import httpx
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alpaca_assets_ingest")

ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
API_KEY = os.environ.get("ALPACA_API_KEY", "").strip()
SECRET = os.environ.get("ALPACA_SECRET_KEY", "").strip()

DSN = os.environ.get("QID_TSDB_DSN") or (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid_analytics')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', '')}"
)


def fetch_assets() -> list[dict]:
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET,
    }
    url = f"{ALPACA_BASE}/v2/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    log.info("fetching %s", url)
    with httpx.Client(timeout=120.0) as c:
        resp = c.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"Alpaca HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    log.info("fetched assets=%d", len(data))
    return data


def normalize(a: dict) -> dict:
    return {
        "symbol":         a["symbol"],
        "name":           a.get("name"),
        "exchange":       a.get("exchange"),
        "asset_class":    a.get("class"),
        "status":         a.get("status"),
        "tradable":       bool(a.get("tradable")),
        "marginable":     bool(a.get("marginable")),
        "shortable":      bool(a.get("shortable")),
        "easy_to_borrow": bool(a.get("easy_to_borrow")),
        "fractionable":   bool(a.get("fractionable")),
        "attributes":     a.get("attributes") or [],
    }


def upsert(rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO alpaca_assets (
            symbol, name, exchange, asset_class, status, tradable,
            marginable, shortable, easy_to_borrow, fractionable, attributes,
            last_refreshed
        ) VALUES (
            %(symbol)s, %(name)s, %(exchange)s, %(asset_class)s, %(status)s, %(tradable)s,
            %(marginable)s, %(shortable)s, %(easy_to_borrow)s, %(fractionable)s, %(attributes)s,
            NOW()
        )
        ON CONFLICT (symbol) DO UPDATE SET
            name = EXCLUDED.name,
            exchange = EXCLUDED.exchange,
            asset_class = EXCLUDED.asset_class,
            status = EXCLUDED.status,
            tradable = EXCLUDED.tradable,
            marginable = EXCLUDED.marginable,
            shortable = EXCLUDED.shortable,
            easy_to_borrow = EXCLUDED.easy_to_borrow,
            fractionable = EXCLUDED.fractionable,
            attributes = EXCLUDED.attributes,
            last_refreshed = NOW()
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        execute_batch(cur, sql, rows, page_size=500)
    return len(rows)


def main() -> int:
    if not API_KEY or not SECRET:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY required")
        return 2
    assets = fetch_assets()
    rows = [normalize(a) for a in assets]
    n = upsert(rows)
    log.info("upserted=%d", n)

    # Coverage report.
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
              COUNT(*) FILTER (WHERE tradable)                AS tradable,
              COUNT(*) FILTER (WHERE tradable AND industry IS NOT NULL) AS with_industry
            FROM alpaca_x_industry
        """)
        tradable, with_ind = cur.fetchone()
        pct = 100 * with_ind / tradable if tradable else 0
        log.info("alpaca_x_industry: tradable=%d with_industry=%d (%.1f%%)",
                 tradable, with_ind, pct)
    return 0


if __name__ == "__main__":
    sys.exit(main())

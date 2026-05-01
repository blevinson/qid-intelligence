#!/usr/bin/env python3
"""Pull FMP profile-bulk → upsert ticker_metadata.

QIDP-226 Layer A. Idempotent. Run as one-shot Job at deploy and weekly
via CronJob to refresh sector/industry/market_cap.

The profile-bulk endpoint is heavily rate-limited (live test: rate-limited
after 4 calls), so this is the *only* way to populate sector lookups
for the full universe — no per-ticker lazy lookups.

Env:
    FMP_API_KEY      — required
    QID_TSDB_DSN     — postgres connection string (or QID_DB_* parts)
    FMP_PARTS        — optional, default '0,1,2,3' (4 chunks of ~22K rows each)
    FMP_INCLUDE_NON_US — '1' to include non-US tickers (default: US-only)
"""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
import time
from typing import Iterator

import httpx
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fmp_bootstrap")

FMP_BASE = "https://financialmodelingprep.com/stable"
API_KEY = os.environ.get("FMP_API_KEY", "").strip()
PARTS = [int(p) for p in os.environ.get("FMP_PARTS", "0,1,2,3").split(",") if p.strip()]
INCLUDE_NON_US = os.environ.get("FMP_INCLUDE_NON_US", "").lower() in ("1", "true", "yes")

DSN = os.environ.get("QID_TSDB_DSN") or (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid_analytics')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', '')}"
)


def fetch_part(part: int) -> list[dict]:
    url = f"{FMP_BASE}/profile-bulk"
    log.info("fetching profile-bulk part=%d", part)
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url, params={"part": part, "apikey": API_KEY})
    if resp.status_code != 200:
        raise RuntimeError(f"FMP HTTP {resp.status_code}: {resp.text[:300]}")
    text = resp.text
    if text.lstrip().startswith("{") and "Error" in text[:200]:
        raise RuntimeError(f"FMP error response: {text[:300]}")
    rows = list(csv.DictReader(io.StringIO(text)))
    log.info("part=%d rows=%d", part, len(rows))
    return rows


def normalize(row: dict) -> dict | None:
    """CSV row → ticker_metadata row, or None to skip."""
    sym = (row.get("symbol") or "").strip()
    if not sym:
        return None
    if not INCLUDE_NON_US and (row.get("country") or "").strip() != "US":
        return None
    if (row.get("isActivelyTrading") or "").lower() != "true":
        return None

    def _bool(v: str | None) -> bool:
        return (v or "").strip().lower() == "true"

    def _int(v: str | None) -> int | None:
        if not v:
            return None
        try:
            return int(float(v))
        except ValueError:
            return None

    return {
        "symbol": sym,
        "name": (row.get("companyName") or "").strip() or None,
        "sector": (row.get("sector") or "").strip() or None,
        "industry": (row.get("industry") or "").strip() or None,
        "market_cap": _int(row.get("marketCap")),
        "exchange": (row.get("exchange") or "").strip() or None,
        "is_etf": _bool(row.get("isEtf")),
        "country": (row.get("country") or "").strip() or None,
    }


def upsert(rows: Iterator[dict]) -> int:
    sql = """
        INSERT INTO ticker_metadata (
            symbol, name, sector, industry, market_cap, exchange, is_etf, country, last_refreshed
        ) VALUES (
            %(symbol)s, %(name)s, %(sector)s, %(industry)s, %(market_cap)s,
            %(exchange)s, %(is_etf)s, %(country)s, NOW()
        )
        ON CONFLICT (symbol) DO UPDATE SET
            name = EXCLUDED.name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            market_cap = EXCLUDED.market_cap,
            exchange = EXCLUDED.exchange,
            is_etf = EXCLUDED.is_etf,
            country = EXCLUDED.country,
            last_refreshed = NOW()
    """
    batch = list(rows)
    if not batch:
        return 0
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            execute_batch(cur, sql, batch, page_size=500)
        conn.commit()
    return len(batch)


def main() -> int:
    if not API_KEY:
        log.error("FMP_API_KEY is required")
        return 1

    total_in = 0
    total_out = 0
    for part in PARTS:
        for attempt in range(3):
            try:
                csv_rows = fetch_part(part)
                break
            except Exception as e:
                log.warning("part=%d attempt=%d failed: %s", part, attempt + 1, e)
                if attempt == 2:
                    log.error("part=%d gave up after 3 attempts", part)
                    raise
                time.sleep(5 * (attempt + 1))
        total_in += len(csv_rows)
        normalized = [n for n in (normalize(r) for r in csv_rows) if n]
        n = upsert(normalized)
        total_out += n
        log.info("part=%d ingested=%d (skipped %d)", part, n, len(csv_rows) - n)

    log.info("done — fetched=%d upserted=%d", total_in, total_out)
    # Print sector breakdown for sanity check
    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT sector, count(*) AS n
                FROM ticker_metadata
                WHERE country = 'US' AND NOT is_etf AND sector IS NOT NULL
                GROUP BY sector ORDER BY n DESC
            """)
            for sector, n in cur.fetchall():
                log.info("  %-30s %d", sector, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())

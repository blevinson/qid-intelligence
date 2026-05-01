#!/usr/bin/env python3
"""QIDP-230 — Alpaca screener movers snapshot → crucix_movers hypertable.

Replaces the FMP movers ingest. Calls three Alpaca screener REST endpoints
and writes a single time-stamped row per (time, category, symbol). Same
PRIMARY KEY (time, category, symbol) deduping as the FMP version.

Endpoints (v1beta1):
  GET https://data.alpaca.markets/v1beta1/screener/stocks/movers?top=50
        → {"gainers":[...], "losers":[...], "market_type":"stocks", ...}
  GET https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=trades&top=50
        → {"most_actives":[{symbol, volume, trade_count}, ...], ...}

Auth: APCA-API-KEY-ID / APCA-API-SECRET-KEY headers.

Env:
    ALPACA_API_KEY     — required
    ALPACA_SECRET_KEY  — required
    ALPACA_DATA_BASE   — default https://data.alpaca.markets
    QID_TSDB_DSN       — postgres connection (or QID_DB_* parts)
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
log = logging.getLogger("alpaca_movers_ingest")

ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "").strip()
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "").strip()
ALPACA_DATA_BASE = os.environ.get(
    "ALPACA_DATA_BASE", "https://data.alpaca.markets"
).rstrip("/")
ALPACA_TOP = int(os.environ.get("ALPACA_MOVERS_TOP", "50"))

DSN = os.environ.get("QID_TSDB_DSN") or (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid_analytics')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', '')}"
)


def _headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "accept": "application/json",
    }


def fetch_movers(client: httpx.Client) -> tuple[list[dict], list[dict]]:
    url = f"{ALPACA_DATA_BASE}/v1beta1/screener/stocks/movers"
    log.info("GET %s top=%d", url, ALPACA_TOP)
    resp = client.get(url, params={"top": ALPACA_TOP})
    if resp.status_code != 200:
        raise RuntimeError(
            f"Alpaca movers HTTP {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    gainers = data.get("gainers") or []
    losers = data.get("losers") or []
    log.info("  gainers=%d losers=%d", len(gainers), len(losers))
    return gainers, losers


def fetch_most_actives(client: httpx.Client) -> list[dict]:
    url = f"{ALPACA_DATA_BASE}/v1beta1/screener/stocks/most-actives"
    log.info("GET %s by=trades top=%d", url, ALPACA_TOP)
    resp = client.get(url, params={"by": "trades", "top": ALPACA_TOP})
    if resp.status_code != 200:
        raise RuntimeError(
            f"Alpaca most-actives HTTP {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    rows = data.get("most_actives") or []
    log.info("  most_actives=%d", len(rows))
    return rows


def normalize_movers(category: str, items: list[dict], ts: datetime) -> list[dict]:
    """Movers payload row shape (per Alpaca v1beta1 screener):
        {symbol, percent_change, change, price}
    `volume` and `trade_count` are not exposed on this endpoint.
    """
    rows: list[dict] = []
    for r in items:
        sym = (r.get("symbol") or "").strip()
        if not sym:
            continue
        rows.append({
            "time": ts,
            "category": category,
            "symbol": sym,
            "name": None,  # screener doesn't expose name; tm join fills it
            "price": r.get("price"),
            "change_pct": r.get("percent_change"),
            "volume": None,
            "trade_count": None,
            "exchange": None,
        })
    return rows


def normalize_most_actives(items: list[dict], ts: datetime) -> list[dict]:
    """Most-actives payload row shape:
        {symbol, volume, trade_count}
    No price/change in the response — accepted; downstream fills via the
    bars/ticker_metadata join.
    """
    rows: list[dict] = []
    for r in items:
        sym = (r.get("symbol") or "").strip()
        if not sym:
            continue
        rows.append({
            "time": ts,
            "category": "most_active",
            "symbol": sym,
            "name": None,
            "price": None,
            "change_pct": None,
            "volume": r.get("volume"),
            "trade_count": r.get("trade_count"),
            "exchange": None,
        })
    return rows


def upsert(rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO crucix_movers
            (time, category, symbol, name, price, change_pct, volume, trade_count, exchange)
        VALUES (%(time)s, %(category)s, %(symbol)s, %(name)s, %(price)s,
                %(change_pct)s, %(volume)s, %(trade_count)s, %(exchange)s)
        ON CONFLICT (time, category, symbol) DO UPDATE SET
            name        = COALESCE(EXCLUDED.name, crucix_movers.name),
            price       = COALESCE(EXCLUDED.price, crucix_movers.price),
            change_pct  = COALESCE(EXCLUDED.change_pct, crucix_movers.change_pct),
            volume      = COALESCE(EXCLUDED.volume, crucix_movers.volume),
            trade_count = COALESCE(EXCLUDED.trade_count, crucix_movers.trade_count),
            exchange    = COALESCE(EXCLUDED.exchange, crucix_movers.exchange)
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        execute_batch(cur, sql, rows, page_size=200)
    return len(rows)


def main() -> int:
    if not ALPACA_KEY or not ALPACA_SECRET:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY required")
        return 2

    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    total = 0
    with httpx.Client(timeout=30.0, headers=_headers()) as client:
        try:
            gainers, losers = fetch_movers(client)
            total += upsert(normalize_movers("gainers", gainers, ts))
            total += upsert(normalize_movers("losers", losers, ts))
        except Exception as e:
            log.warning("movers fetch failed: %s", e)

        try:
            ma = fetch_most_actives(client)
            total += upsert(normalize_most_actives(ma, ts))
        except Exception as e:
            log.warning("most-actives fetch failed: %s", e)

    if total == 0:
        log.error("ingested 0 rows — refusing to claim success")
        return 1

    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT category,
                   COUNT(*) AS n,
                   COUNT(*) FILTER (WHERE trade_count IS NOT NULL) AS with_trades,
                   COUNT(*) FILTER (WHERE volume IS NOT NULL) AS with_volume
            FROM crucix_movers WHERE time = %s
            GROUP BY 1 ORDER BY 1
            """,
            (ts,),
        )
        for cat, n, with_trades, with_volume in cur.fetchall():
            log.info(
                "  %s: rows=%d trade_count_filled=%d volume_filled=%d",
                cat, n, with_trades, with_volume,
            )
    log.info("total upserted=%d at ts=%s", total, ts.isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pull Alpaca daily OHLCV bars → MinIO parquet, one file per symbol.

QIDP-229 Sprint 1 — equity backtest data foundation. Idempotent append:
re-runs only fetch bars from (max_dt_in_parquet + 1) to today.

Layout: s3://qid-equity/bars/ohlcv-1d/{symbol}.parquet — full history per
symbol. Mirrors tradefarm's data/minio_cache.py pattern (equities/ohlcv-1d/).

Universe sourced from alpaca_x_industry view (TSDB) filtered to
mid+large equity (market_cap >= $2B) plus sector ETFs by symbol allowlist.

Env:
    ALPACA_API_KEY, ALPACA_SECRET_KEY  — required
    ALPACA_BASE_DATA_URL               — optional, default https://data.alpaca.markets
    QID_TSDB_DSN | QID_DB_*            — for universe SQL
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY — required
    MINIO_BUCKET                       — default qid-equity
    BARS_START_DATE                    — optional ISO date for cold start; default 5y back from today
    BARS_BATCH_SIZE                    — symbols per Alpaca call, default 100 (Alpaca's docs cap at 200)
    BARS_SYMBOLS                       — optional CSV override universe; useful for smoke runs
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd
import psycopg2
from minio import Minio
from minio.error import S3Error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("alpaca_bars_ingest")

ALPACA_BASE = os.environ.get("ALPACA_BASE_DATA_URL", "https://data.alpaca.markets").rstrip("/")
API_KEY = os.environ.get("ALPACA_API_KEY", "").strip()
SECRET = os.environ.get("ALPACA_SECRET_KEY", "").strip()
BATCH_SIZE = int(os.environ.get("BARS_BATCH_SIZE", "100"))

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio.qid.svc.cluster.local:9000")
MINIO_ACCESS = os.environ.get("MINIO_ACCESS_KEY", "").strip()
MINIO_SECRET = os.environ.get("MINIO_SECRET_KEY", "").strip()
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "qid-equity")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")

DSN = os.environ.get("QID_TSDB_DSN") or (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid_analytics')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', '')}"
)

OBJECT_PREFIX = "bars/ohlcv-1d"
DEFAULT_LOOKBACK_DAYS = 365 * 5  # 5y cold start


def _alpaca_headers() -> dict[str, str]:
    return {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET}


def fetch_universe() -> list[str]:
    """Mid+large cap stocks + sector ETFs from alpaca_x_industry."""
    override = os.environ.get("BARS_SYMBOLS", "").strip()
    if override:
        symbols = sorted({s.strip().upper() for s in override.split(",") if s.strip()})
        log.info("BARS_SYMBOLS override → %d symbols", len(symbols))
        return symbols

    sql = """
        SELECT a.symbol
        FROM alpaca_assets a
        JOIN ticker_metadata m ON m.symbol = a.symbol
        WHERE a.tradable
          AND a.asset_class = 'us_equity'
          AND a.exchange IN ('NASDAQ','NYSE','AMEX','ARCA')
          AND m.market_cap >= 2000000000
        ORDER BY 1
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql)
        symbols = [r[0] for r in cur.fetchall()]
    log.info("universe → %d symbols (mid+large)", len(symbols))
    return symbols


def _minio() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE)


def _key(symbol: str) -> str:
    return f"{OBJECT_PREFIX}/{symbol.upper()}.parquet"


def read_existing(client: Minio, symbol: str) -> pd.DataFrame | None:
    try:
        resp = client.get_object(MINIO_BUCKET, _key(symbol))
        data = resp.read()
        resp.close()
        resp.release_conn()
        return pd.read_parquet(io.BytesIO(data))
    except S3Error as e:
        if e.code == "NoSuchKey":
            return None
        raise


def write_parquet(client: Minio, symbol: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    buf.seek(0)
    client.put_object(
        MINIO_BUCKET, _key(symbol), buf, length=len(buf.getvalue()),
        content_type="application/octet-stream",
    )


def fetch_bars(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for a batch of symbols. Handles pagination."""
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    url = f"{ALPACA_BASE}/v2/stocks/bars"
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "raw",
        "feed": "sip",
        "limit": 10000,
    }
    feed_name = params["feed"]
    page = 0
    with httpx.Client(timeout=60.0) as c:
        while True:
            resp = c.get(url, headers=_alpaca_headers(), params=params)
            if resp.status_code != 200:
                # Fail loud. Silent fallback to IEX hides train/serve drift —
                # IEX volume is ~3% of consolidated flow, useless for any
                # flow-based signal. If SIP entitlement lapses, we want to
                # know and pause, not pretend.
                raise RuntimeError(
                    f"Alpaca HTTP {resp.status_code} on feed={feed_name}: {resp.text[:300]}"
                )
            j = resp.json()
            page += 1
            for sym, bars in (j.get("bars") or {}).items():
                if bars:
                    out[sym].extend(bars)
            tok = j.get("next_page_token")
            if not tok:
                break
            params["page_token"] = tok
            time.sleep(0.05)  # gentle rate-pacing

    frames: dict[str, pd.DataFrame] = {}
    for sym, rows in out.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "t": "ts", "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "n": "trade_count", "vw": "vwap",
        })
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["dt"] = df["ts"].dt.date
        df = df[["dt", "open", "high", "low", "close", "volume", "trade_count", "vwap"]]
        df = df.drop_duplicates(subset=["dt"]).sort_values("dt").reset_index(drop=True)
        frames[sym] = df
    return frames


def merge_and_write(client: Minio, symbol: str, new_df: pd.DataFrame) -> tuple[int, int]:
    """Append new_df to existing parquet (idempotent). Returns (added, total)."""
    existing = read_existing(client, symbol)
    if existing is None or existing.empty:
        write_parquet(client, symbol, new_df)
        return len(new_df), len(new_df)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    added = len(combined) - len(existing)
    if added > 0:
        write_parquet(client, symbol, combined)
    return added, len(combined)


def latest_dt(client: Minio, symbol: str) -> date | None:
    df = read_existing(client, symbol)
    if df is None or df.empty:
        return None
    return df["dt"].max()


def main() -> int:
    if not API_KEY or not SECRET:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY required")
        return 2
    if not MINIO_ACCESS or not MINIO_SECRET:
        log.error("MINIO_ACCESS_KEY / MINIO_SECRET_KEY required")
        return 2

    client = _minio()
    if not client.bucket_exists(MINIO_BUCKET):
        log.error("bucket %s does not exist; create it first", MINIO_BUCKET)
        return 2

    symbols = fetch_universe()
    today = date.today()
    cold_start = date.today() - timedelta(days=int(os.environ.get("BARS_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)))
    if os.environ.get("BARS_START_DATE"):
        cold_start = date.fromisoformat(os.environ["BARS_START_DATE"])

    total_added = 0
    skipped = 0
    failed = 0

    # Group symbols into batches
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]

        # Per-batch start = max of (oldest cold_start, latest existing+1 across batch)
        # Keep it simple: use cold_start always; merge_and_write handles dedup.
        # For the daily-refresh CronJob, set BARS_LOOKBACK_DAYS=7 so we only
        # pull last week.
        start = cold_start
        end = today + timedelta(days=1)  # exclusive end → include today's bar if available

        try:
            frames = fetch_bars(batch, start=start, end=end)
        except Exception as e:
            log.error("batch %d-%d failed: %s", i, i + len(batch), e)
            failed += len(batch)
            continue

        for sym in batch:
            df = frames.get(sym)
            if df is None or df.empty:
                skipped += 1
                continue
            try:
                added, total = merge_and_write(client, sym, df)
                total_added += added
                if added > 0:
                    log.info("  %s +%d rows (total=%d, dt range %s..%s)",
                             sym, added, total, df["dt"].min(), df["dt"].max())
            except Exception as e:
                log.error("write %s failed: %s", sym, e)
                failed += 1

        log.info("batch %d-%d done (added=%d skipped=%d failed=%d so far)",
                 i, i + len(batch), total_added, skipped, failed)
        time.sleep(0.2)  # inter-batch cool-down

    log.info("done: symbols=%d total_added=%d skipped=%d failed=%d",
             len(symbols), total_added, skipped, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

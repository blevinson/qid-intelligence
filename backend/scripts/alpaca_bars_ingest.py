#!/usr/bin/env python3
"""Pull daily OHLCV bars → MinIO parquet, one file per symbol.

FIN-3442 follow-up: source = SHARADAR (Nasdaq Data Link), NOT Alpaca. Alpaca's
basic plan only serves recent IEX bars (~3% of consolidated volume — useless for
flow signals) and 403s on recent SIP without an Algo-Trader-Plus subscription.
We already pay for the Sharadar Core US Equities bundle, which gives REAL
consolidated daily OHLCV (incl. delisted names). SEP = stocks, SFP = funds/ETFs.

Layout: s3://qid-equity/bars/ohlcv-1d/{symbol}.parquet — full history per symbol.
Idempotent append: merge_and_write dedups on dt, so re-runs only add new bars.
Raw (unadjusted) close, matching the prior adjustment=raw contract.

Env:
    NASDAQ_DATA_LINK_API_KEY            — required (Sharadar)
    QID_TSDB_DSN | QID_DB_*            — for universe SQL
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY — required
    MINIO_BUCKET                       — default qid-equity
    BARS_START_DATE / BARS_LOOKBACK_DAYS — cold-start window (default 5y); the
                                         daily CronJob sets BARS_LOOKBACK_DAYS=7
    BARS_BATCH_SIZE                    — tickers per Sharadar call, default 100
    BARS_SYMBOLS                       — optional CSV override universe (smoke runs)

Deploy (qid ns): this runs from ConfigMap ``alpaca-bars-ingest-script`` (key
``alpaca_bars_ingest.py``), mounted at /scripts by the ``alpaca-bars-ingest``
CronJob — NOT baked into the bridge image. This file is the version-controlled
source of truth; to push a change to prod:
    kubectl -n qid create configmap alpaca-bars-ingest-script \\
      --from-file=alpaca_bars_ingest.py=backend/scripts/alpaca_bars_ingest.py \\
      --dry-run=client -o yaml | kubectl -n qid apply -f -
The CronJob needs NASDAQ_DATA_LINK_API_KEY (from secret ``nasdaq-data-link``).
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from datetime import date, timedelta

import httpx
import pandas as pd
import psycopg2
from minio import Minio
from minio.error import S3Error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Sharadar takes the api_key as a query param; httpx logs full URLs at INFO,
# which would leak the key into pod logs. Quiet httpx/httpcore to WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("bars_ingest")

NDL_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
NDL_API_KEY = os.environ.get("NASDAQ_DATA_LINK_API_KEY", "").strip()
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


def fetch_universe() -> list[str]:
    """Mid+large cap stocks + sector ETFs from the assets/metadata tables."""
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


def _fetch_table(table: str, symbols: list[str], start: date, end: date) -> dict[str, list[dict]]:
    """Page through one Sharadar datatable (SEP|SFP) for a batch of tickers."""
    rows_by_sym: dict[str, list[dict]] = {}
    params: dict[str, str] = {
        "ticker": ",".join(symbols),
        "date.gte": start.isoformat(),
        "date.lte": (end - timedelta(days=1)).isoformat(),
        "qopts.columns": "ticker,date,open,high,low,close,volume",
        "api_key": NDL_API_KEY,
    }
    url = f"{NDL_BASE}/{table}.json"
    with httpx.Client(timeout=60.0) as c:
        while True:
            resp = c.get(url, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"Sharadar {table} HTTP {resp.status_code}: {resp.text[:300]}")
            j = resp.json()
            dt = j.get("datatable") or {}
            cols = [col["name"] for col in dt.get("columns", [])]
            for row in dt.get("data", []):
                rec = dict(zip(cols, row))
                rows_by_sym.setdefault(rec.get("ticker"), []).append(rec)
            cursor = (j.get("meta") or {}).get("next_cursor_id")
            if not cursor:
                break
            params["qopts.cursor_id"] = cursor
            time.sleep(0.05)
    return rows_by_sym


def fetch_bars(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Daily bars for a batch from Sharadar — SEP (stocks) + SFP (funds/ETFs).

    REAL consolidated volume. Raw (unadjusted) close. No per-bar trade_count/vwap
    in the Sharadar daily tables — kept as NA so the parquet schema is stable.
    """
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    for table in ("SEP", "SFP"):  # SEP=stocks, SFP=funds/ETFs
        for sym, rows in _fetch_table(table, symbols, start, end).items():
            if sym in out:
                out[sym].extend(rows)

    frames: dict[str, pd.DataFrame] = {}
    for sym, rows in out.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).rename(columns={"date": "dt"})
        df["dt"] = pd.to_datetime(df["dt"]).dt.date
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trade_count"] = pd.NA
        df["vwap"] = pd.NA
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
    combined = combined.drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)
    added = len(combined) - len(existing)
    if added > 0:
        write_parquet(client, symbol, combined)
    return added, len(combined)


def main() -> int:
    if not NDL_API_KEY:
        log.error("NASDAQ_DATA_LINK_API_KEY required")
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

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        start = cold_start
        end = today + timedelta(days=1)  # exclusive end → include today's bar if available

        try:
            frames = fetch_bars(batch, start=start, end=end)
        except Exception as e:  # noqa: BLE001
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
            except Exception as e:  # noqa: BLE001
                log.error("write %s failed: %s", sym, e)
                failed += 1

        log.info("batch %d-%d done (added=%d skipped=%d failed=%d so far)",
                 i, i + len(batch), total_added, skipped, failed)
        time.sleep(0.2)

    log.info("done: symbols=%d total_added=%d skipped=%d failed=%d",
             len(symbols), total_added, skipped, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

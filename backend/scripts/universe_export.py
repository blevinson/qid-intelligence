#!/usr/bin/env python3
"""Export the breakout-universe snapshot to MinIO.

QIDP-229 Sprint 1 — let tradefarm read the symbol list without a direct
TSDB dependency. One JSON object per run, overwrites
qid-equity/universe/snapshot.json.

Universe rule: tradable US equity OR sector ETF, listed on a major exchange,
market_cap >= $2B (or known sector ETF allowlist).

Env:
    MINIO_*           — endpoint, access/secret, bucket (default qid-equity)
    QID_DB_*          — postgres connection
    UNIVERSE_MIN_CAP  — float, default 2e9
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import datetime, timezone

import psycopg2
from minio import Minio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("universe_export")

MIN_CAP = float(os.environ.get("UNIVERSE_MIN_CAP", "2000000000"))

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

OBJECT_KEY = "universe/snapshot.json"

# Hand-curated sector / theme ETFs we always want in the breakout universe,
# even if their market cap drifts under the $2B threshold (e.g. niche themes
# crucix may flag).
ETF_ALLOWLIST = {
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # 11 GICS sector SPDRs
    "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    # Theme / catalyst-relevant
    "ITA", "USO", "UCO", "BNO", "UNG", "GLD", "SLV", "TLT", "HYG", "LQD",
    "URA", "SMH", "ARKK", "FXI", "KWEB", "EWZ", "EWJ", "EWG",
    "UVXY", "VXX",
}


def fetch_universe() -> list[dict]:
    sql = """
        SELECT
            a.symbol,
            COALESCE(a.name, m.name) AS name,
            m.sector,
            m.industry,
            m.market_cap,
            a.exchange,
            COALESCE(m.is_etf, FALSE) AS is_etf,
            a.tradable
        FROM alpaca_assets a
        LEFT JOIN ticker_metadata m ON m.symbol = a.symbol
        WHERE a.tradable
          AND a.asset_class = 'us_equity'
          AND a.exchange IN ('NASDAQ','NYSE','AMEX','ARCA')
          AND (
              m.market_cap >= %s
              OR a.symbol = ANY(%s)
          )
        ORDER BY a.symbol
    """
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, (MIN_CAP, list(ETF_ALLOWLIST)))
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    log.info("universe → %d rows", len(rows))
    return rows


def write_snapshot(rows: list[dict]) -> int:
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter": {
            "min_market_cap_usd": MIN_CAP,
            "exchanges": ["NASDAQ", "NYSE", "AMEX", "ARCA"],
            "etf_allowlist": sorted(ETF_ALLOWLIST),
        },
        "count": len(rows),
        "symbols": rows,
    }
    body = json.dumps(payload, default=str, indent=2).encode("utf-8")
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE)
    if not client.bucket_exists(MINIO_BUCKET):
        raise RuntimeError(f"bucket {MINIO_BUCKET} does not exist")
    client.put_object(
        MINIO_BUCKET, OBJECT_KEY, io.BytesIO(body), length=len(body),
        content_type="application/json",
    )
    return len(body)


def main() -> int:
    if not MINIO_ACCESS or not MINIO_SECRET:
        log.error("MINIO_ACCESS_KEY / MINIO_SECRET_KEY required")
        return 2
    rows = fetch_universe()
    if not rows:
        log.error("empty universe — refusing to overwrite snapshot")
        return 1
    n_bytes = write_snapshot(rows)
    log.info("wrote s3://%s/%s (%d rows, %d bytes)", MINIO_BUCKET, OBJECT_KEY, len(rows), n_bytes)

    # Quick stats
    by_sector: dict[str, int] = {}
    n_etf = 0
    for r in rows:
        sec = r["sector"] or "(unknown)"
        by_sector[sec] = by_sector.get(sec, 0) + 1
        if r["is_etf"]:
            n_etf += 1
    log.info("breakdown: stocks=%d  etfs=%d", len(rows) - n_etf, n_etf)
    for sec, n in sorted(by_sector.items(), key=lambda kv: -kv[1]):
        log.info("  %-25s %d", sec, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Finnhub earnings-calendar ingest into crucix_earnings_calendar.
Pulls forward-looking earnings dates (today + 60 days) and upserts on
(symbol, earnings_date). Filters to panel universe via ticker_metadata.
Designed to run daily as a k8s CronJob (06:00 UTC).
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import requests

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"

TSDB_DSN = os.environ.get("QID_TSDB_DSN", (
    f"host={os.environ.get('QID_DB_HOST', 'timescaledb.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', 'qid')}"
))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS crucix_earnings_calendar (
  symbol            TEXT NOT NULL,
  earnings_date     DATE NOT NULL,
  hour              TEXT,
  eps_estimate      DOUBLE PRECISION,
  revenue_estimate  DOUBLE PRECISION,
  year              INT,
  quarter           INT,
  fetched_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (symbol, earnings_date)
);

SELECT create_hypertable('crucix_earnings_calendar', 'earnings_date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_cec_date ON crucix_earnings_calendar(earnings_date);
CREATE INDEX IF NOT EXISTS idx_cec_symbol ON crucix_earnings_calendar(symbol, earnings_date DESC);
"""

FORWARD_DAYS = 60


def load_panel_symbols(conn):
    """Load the set of symbols tracked in ticker_metadata."""
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM ticker_metadata")
        return {row[0] for row in cur.fetchall()}


def fetch_earnings(date_from, date_to):
    resp = requests.get(
        FINNHUB_EARNINGS_URL,
        params={
            "from": date_from,
            "to": date_to,
            "token": FINNHUB_API_KEY,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("earningsCalendar", [])


def parse_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_int(val):
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def upsert_earnings(conn, records, panel_symbols):
    filtered = [
        r for r in records
        if r.get("symbol") in panel_symbols and r.get("date")
    ]
    if not filtered:
        print("[finnhub_earnings_ingest] No panel-universe earnings to upsert")
        return 0

    values = [
        (
            r["symbol"],
            r["date"],
            r.get("hour"),
            parse_float(r.get("epsEstimate")),
            parse_float(r.get("revenueEstimate")),
            parse_int(r.get("year")),
            parse_int(r.get("quarter")),
        )
        for r in filtered
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO crucix_earnings_calendar
               (symbol, earnings_date, hour, eps_estimate, revenue_estimate,
                year, quarter, fetched_at)
               VALUES %s
               ON CONFLICT (symbol, earnings_date) DO UPDATE SET
                 hour             = EXCLUDED.hour,
                 eps_estimate     = EXCLUDED.eps_estimate,
                 revenue_estimate = EXCLUDED.revenue_estimate,
                 year             = EXCLUDED.year,
                 quarter          = EXCLUDED.quarter,
                 fetched_at       = NOW()""",
            values,
            template="(%s, %s, %s, %s, %s, %s, %s, NOW())",
        )
        conn.commit()

    print(f"[finnhub_earnings_ingest] Upserted {len(filtered)} earnings ({len(records) - len(filtered)} filtered out)")
    return len(filtered)


def main():
    if not FINNHUB_API_KEY:
        print("[finnhub_earnings_ingest] ERROR: FINNHUB_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(TSDB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()

        panel_symbols = load_panel_symbols(conn)
        print(f"[finnhub_earnings_ingest] Panel universe: {len(panel_symbols)} symbols")

        today = datetime.now(timezone.utc).date()
        date_from = today.isoformat()
        date_to = (today + timedelta(days=FORWARD_DAYS)).isoformat()

        print(f"[finnhub_earnings_ingest] Fetching earnings calendar {date_from} to {date_to}")
        records = fetch_earnings(date_from, date_to)
        print(f"[finnhub_earnings_ingest] Fetched {len(records)} raw earnings from Finnhub")

        count = upsert_earnings(conn, records, panel_symbols)
        print(f"[finnhub_earnings_ingest] Done — {count} rows upserted")

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM crucix_earnings_calendar")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(DISTINCT symbol) FROM crucix_earnings_calendar WHERE earnings_date >= %s",
                (today.isoformat(),),
            )
            symbols = cur.fetchone()[0]
            print(f"[finnhub_earnings_ingest] Table totals: {total} rows, {symbols} distinct symbols (upcoming)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

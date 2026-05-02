#!/usr/bin/env python3
"""
SEC EDGAR CIK → ticker lookup table ingest.
Pulls https://www.sec.gov/files/company_tickers.json and upserts into sec_cik_ticker.
Designed to run as a daily k8s CronJob.
"""

import json
import os
import sys
import time

import psycopg2
import psycopg2.extras
import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "FinTorch crucix-ingest brantjlevinson@gmail.com"

TSDB_DSN = os.environ.get("QID_TSDB_DSN", (
    f"host={os.environ.get('QID_DB_HOST', 'timescaledb.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', 'qid')}"
))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sec_cik_ticker (
  cik         BIGINT PRIMARY KEY,
  ticker      TEXT NOT NULL,
  company     TEXT,
  fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sec_cik_ticker ON sec_cik_ticker(ticker);
"""

UPSERT_SQL = """
INSERT INTO sec_cik_ticker (cik, ticker, company, fetched_at)
VALUES %s
ON CONFLICT (cik) DO UPDATE SET
  ticker     = EXCLUDED.ticker,
  company    = EXCLUDED.company,
  fetched_at = EXCLUDED.fetched_at
"""


def fetch_tickers():
    resp = requests.get(SEC_TICKERS_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    seen = {}
    for entry in data.values():
        cik = int(entry["cik_str"])
        ticker = entry["ticker"].upper()
        company = entry.get("title", "")
        seen[cik] = (cik, ticker, company)
    return list(seen.values())


def main():
    print(f"[sec_cik_ticker_ingest] Fetching {SEC_TICKERS_URL}")
    rows = fetch_tickers()
    print(f"[sec_cik_ticker_ingest] Parsed {len(rows)} CIK→ticker mappings")

    conn = psycopg2.connect(TSDB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()

            template = "(%(cik)s, %(ticker)s, %(company)s, NOW())"
            values = [{"cik": r[0], "ticker": r[1], "company": r[2]} for r in rows]

            batch_size = 2000
            total_upserted = 0
            for i in range(0, len(values), batch_size):
                batch = values[i : i + batch_size]
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO sec_cik_ticker (cik, ticker, company, fetched_at)
                       VALUES %s
                       ON CONFLICT (cik) DO UPDATE SET
                         ticker     = EXCLUDED.ticker,
                         company    = EXCLUDED.company,
                         fetched_at = EXCLUDED.fetched_at""",
                    [(v["cik"], v["ticker"], v["company"]) for v in batch],
                    template="(%s, %s, %s, NOW())",
                    page_size=batch_size,
                )
                total_upserted += len(batch)
            conn.commit()

            cur.execute("SELECT COUNT(*) FROM sec_cik_ticker")
            count = cur.fetchone()[0]
            print(f"[sec_cik_ticker_ingest] Upserted {total_upserted} rows, table now has {count} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

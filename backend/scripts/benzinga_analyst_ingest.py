#!/usr/bin/env python3
"""
Benzinga analyst-actions ingest into crucix_analyst_actions.
Pulls upgrades, downgrades, price-target changes, initiations from
Benzinga's calendar/ratings endpoint. Upserts on benzinga_id for idempotency.
Designed to run 3x/day as a k8s CronJob.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import requests

BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY", "")
BENZINGA_RATINGS_URL = (
    "https://api.benzinga.com/api/v2.1/calendar/ratings"
    "?token={token}&parameters[date_from]={date_from}&parameters[date_to]={date_to}"
    "&pagesize=1000&page={page}"
)

TSDB_DSN = os.environ.get("QID_TSDB_DSN", (
    f"host={os.environ.get('QID_DB_HOST', 'timescaledb.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', 'qid')}"
))

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS crucix_analyst_actions (
  action_date     DATE NOT NULL,
  symbol          TEXT NOT NULL,
  benzinga_id     TEXT NOT NULL,
  action_company  TEXT,
  action_pt       TEXT,
  rating_current  TEXT,
  rating_prior    TEXT,
  pt_current      DOUBLE PRECISION,
  pt_prior        DOUBLE PRECISION,
  pt_delta_pct    DOUBLE PRECISION,
  importance      INT,
  notes           TEXT,
  fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (benzinga_id)
);

SELECT create_hypertable('crucix_analyst_actions', 'action_date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_caa_symbol_date ON crucix_analyst_actions(symbol, action_date DESC);
CREATE INDEX IF NOT EXISTS idx_caa_action_date ON crucix_analyst_actions(action_date DESC);
"""

MAX_PAGES = 50


def load_panel_symbols(conn):
    """Load the set of symbols tracked in ticker_metadata."""
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM ticker_metadata")
        return {row[0] for row in cur.fetchall()}


def fetch_ratings_page(date_from, date_to, page):
    url = BENZINGA_RATINGS_URL.format(
        token=BENZINGA_API_KEY,
        date_from=date_from,
        date_to=date_to,
        page=page,
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def compute_pt_delta_pct(pt_current, pt_prior):
    if pt_current is None or pt_prior is None or pt_prior == 0:
        return None
    return (pt_current - pt_prior) / pt_prior * 100.0


def fetch_all_ratings(date_from, date_to):
    all_actions = []
    for page in range(MAX_PAGES):
        print(f"[benzinga_analyst_ingest] Fetching page {page} ({date_from} to {date_to})")
        data = fetch_ratings_page(date_from, date_to, page)

        ratings = data if isinstance(data, list) else data.get("ratings", data.get("data", []))
        if not ratings:
            break

        for r in ratings:
            pt_cur = parse_float(r.get("pt_current") or r.get("adjusted_pt_current"))
            pt_pri = parse_float(r.get("pt_prior") or r.get("adjusted_pt_prior"))

            all_actions.append({
                "action_date": r.get("date"),
                "symbol": r.get("ticker", "").upper(),
                "benzinga_id": str(r.get("id", "")),
                "action_company": r.get("analyst") or r.get("analyst_name"),
                "action_pt": r.get("action_pt"),
                "rating_current": r.get("rating_current"),
                "rating_prior": r.get("rating_prior"),
                "pt_current": pt_cur,
                "pt_prior": pt_pri,
                "pt_delta_pct": compute_pt_delta_pct(pt_cur, pt_pri),
                "importance": int(r["importance"]) if r.get("importance") is not None else None,
                "notes": r.get("notes") or r.get("description"),
            })

        if len(ratings) < 1000:
            break

    return all_actions


def upsert_actions(conn, actions, panel_symbols):
    filtered = [a for a in actions if a["symbol"] in panel_symbols and a["benzinga_id"]]
    if not filtered:
        print("[benzinga_analyst_ingest] No panel-universe actions to upsert")
        return 0

    with conn.cursor() as cur:
        values = [
            (
                a["action_date"], a["symbol"], a["benzinga_id"],
                a["action_company"], a["action_pt"],
                a["rating_current"], a["rating_prior"],
                a["pt_current"], a["pt_prior"], a["pt_delta_pct"],
                a["importance"], a["notes"],
            )
            for a in filtered
        ]
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO crucix_analyst_actions
               (action_date, symbol, benzinga_id, action_company, action_pt,
                rating_current, rating_prior, pt_current, pt_prior, pt_delta_pct,
                importance, notes, fetched_at)
               VALUES %s
               ON CONFLICT (benzinga_id) DO UPDATE SET
                 action_company = EXCLUDED.action_company,
                 action_pt      = EXCLUDED.action_pt,
                 rating_current = EXCLUDED.rating_current,
                 rating_prior   = EXCLUDED.rating_prior,
                 pt_current     = EXCLUDED.pt_current,
                 pt_prior       = EXCLUDED.pt_prior,
                 pt_delta_pct   = EXCLUDED.pt_delta_pct,
                 importance     = EXCLUDED.importance,
                 notes          = EXCLUDED.notes,
                 fetched_at     = NOW()""",
            values,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
        )
        conn.commit()

    print(f"[benzinga_analyst_ingest] Upserted {len(filtered)} actions ({len(actions) - len(filtered)} filtered out)")
    return len(filtered)


def main():
    if not BENZINGA_API_KEY:
        print("[benzinga_analyst_ingest] ERROR: BENZINGA_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(TSDB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()

        panel_symbols = load_panel_symbols(conn)
        print(f"[benzinga_analyst_ingest] Panel universe: {len(panel_symbols)} symbols")

        today = datetime.now(timezone.utc).date()
        date_from = (today - timedelta(days=3)).isoformat()
        date_to = today.isoformat()

        actions = fetch_all_ratings(date_from, date_to)
        print(f"[benzinga_analyst_ingest] Fetched {len(actions)} raw ratings from Benzinga")

        count = upsert_actions(conn, actions, panel_symbols)
        print(f"[benzinga_analyst_ingest] Done — {count} rows upserted")

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM crucix_analyst_actions")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(DISTINCT symbol) FROM crucix_analyst_actions WHERE action_date >= %s",
                (date_from,),
            )
            symbols = cur.fetchone()[0]
            print(f"[benzinga_analyst_ingest] Table totals: {total} actions, {symbols} distinct symbols (last 3d)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

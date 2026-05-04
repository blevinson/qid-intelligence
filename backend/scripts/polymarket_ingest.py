#!/usr/bin/env python3
"""
Polymarket geopolitical-theme price snapshot ingest.

Phase 1 (QIDP-242 / FIN-1108): read-only ingest from public Polymarket Gamma API.
NO wallet, NO authentication, NO trading.

Tables created/populated:
  - polymarket_markets       (market metadata + resolution text)
  - polymarket_price_snap    (TimescaleDB hypertable, hourly snapshots)
  - polymarket_position_snap (TimescaleDB hypertable, top-10 wallet positions)

Hourly k8s CronJob in qid namespace. Mirrors FIN-1070/1071/1072 UPSERT conventions.
"""

import os
import sys
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import requests

TAG = "[polymarket_ingest]"

# ── Public Polymarket API endpoints (no auth required) ────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
PAGE_SIZE = 100
MAX_ACTIVE_PAGES = 20   # 2000 active markets max
MAX_RESOLVED_PAGES = 5  # 500 recent resolved markets
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.25    # polite throttle between pages

# ── Database connection ───────────────────────────────────────────────────────
TSDB_DSN = os.environ.get("QID_TSDB_DSN", (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid')} "
    f"user={os.environ.get('QID_DB_USER', 'qid')} "
    f"password={os.environ.get('QID_DB_PASS', 'qid')}"
))

# ── Theme keyword filter (v1: dumb substring match — over-ingest, filter later)
THEME_KEYWORDS = [
    # Middle East
    "hormuz", "israel", "iran", "lebanon", "hezbollah", "houthi",
    # Asia
    "taiwan", "korea", "china", "russia", "ukraine",
    # US policy
    "federal reserve", "fed rate", "interest rate", "election",
    "sanction", "tariff", "presidential", "executive order",
    # Energy / commodities
    "brent", "wti", "opec", "crude oil", "oil supply", "oil price", "gold",
    # Defense / security
    "nato", "arms deal", "military", "invasion", "ceasefire",
    "nuclear", "missile", "conflict",
    # Generic geopolitical
    "escalat", "coup", "regime change",
]


def matches_theme(question: str) -> list:
    q_lower = question.lower()
    return [kw for kw in THEME_KEYWORDS if kw in q_lower]


# ── DDL ───────────────────────────────────────────────────────────────────────
INIT_SQL = """
CREATE TABLE IF NOT EXISTS polymarket_markets (
    market_id           TEXT PRIMARY KEY,
    question            TEXT NOT NULL,
    theme_tags          TEXT[],
    end_date            DATE,
    status              TEXT,
    created_at          TIMESTAMPTZ,
    resolved_at         TIMESTAMPTZ,
    resolved_outcome    TEXT,
    resolution_text     TEXT,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS polymarket_price_snap (
    market_id   TEXT             NOT NULL,
    time        TIMESTAMPTZ      NOT NULL,
    bid         DOUBLE PRECISION,
    ask         DOUBLE PRECISION,
    mid         DOUBLE PRECISION,
    last        DOUBLE PRECISION,
    volume_24h  DOUBLE PRECISION,
    PRIMARY KEY (market_id, time)
);

SELECT create_hypertable('polymarket_price_snap', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_pps_market_time
    ON polymarket_price_snap (market_id, time DESC);

CREATE TABLE IF NOT EXISTS polymarket_position_snap (
    market_id               TEXT        NOT NULL,
    time                    TIMESTAMPTZ NOT NULL,
    side                    TEXT        NOT NULL,
    wallet_id               TEXT        NOT NULL,
    position_usd            DOUBLE PRECISION,
    wallet_lifetime_winrate DOUBLE PRECISION,
    PRIMARY KEY (market_id, time, side, wallet_id)
);

SELECT create_hypertable('polymarket_position_snap', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ppos_market_time
    ON polymarket_position_snap (market_id, time DESC);
"""


# ── API helpers ───────────────────────────────────────────────────────────────

def fetch_markets_page(offset, active):
    url = (
        f"{GAMMA_API}/markets"
        f"?limit={PAGE_SIZE}&offset={offset}"
        f"&active={'true' if active else 'false'}"
        f"&order=volume24hr&ascending=false"
    )
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else data.get("markets", [])


def fetch_all_theme_markets():
    matched = []
    seen_ids = set()

    active_configs = [
        (True, MAX_ACTIVE_PAGES),
        (False, MAX_RESOLVED_PAGES),
    ]
    for active, max_pages in active_configs:
        label = "active" if active else "resolved"
        for page in range(max_pages):
            offset = page * PAGE_SIZE
            print(f"{TAG} Fetching {label} markets page {page} (offset {offset})")
            try:
                markets = fetch_markets_page(offset, active)
            except requests.RequestException as exc:
                print(f"{TAG} WARNING: fetch error at {label} page {page}: {exc}", file=sys.stderr)
                break

            if not markets:
                break

            for m in markets:
                mid = m.get("conditionId") or m.get("id")
                if not mid or mid in seen_ids:
                    continue
                question = m.get("question", "")
                tags = matches_theme(question)
                if tags:
                    m["_market_id"] = mid
                    m["_matched_tags"] = tags
                    matched.append(m)
                    seen_ids.add(mid)

            if len(markets) < PAGE_SIZE:
                break

            time.sleep(REQUEST_DELAY)

    print(f"{TAG} Theme-matched markets: {len(matched)}")
    return matched


def fetch_top_positions(market_id):
    """
    Attempt top-10 wallet positions via public data API.
    Returns None if endpoint requires auth (401/403) or fails.
    """
    try:
        url = (
            f"{DATA_API}/positions"
            f"?market={market_id}&sizeThreshold=0&limit=10&sortBy=size&order=DESC"
        )
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code in (401, 403):
            return None
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("positions", [])
    except requests.RequestException:
        return None


def parse_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_ts(val):
    if not val:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(val, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(val)
    except ValueError:
        return None


def end_date_str(m):
    raw = m.get("endDate", "") or ""
    return raw[:10] if raw else None


# ── UPSERT helpers ────────────────────────────────────────────────────────────

def upsert_markets(conn, markets):
    rows = [
        (
            m["_market_id"],
            m.get("question", ""),
            m["_matched_tags"],
            end_date_str(m),
            "resolved" if m.get("resolved") else ("closed" if m.get("closed") else "active"),
            parse_ts(m.get("startDate")),
            parse_ts(m.get("resolutionTime") or m.get("resolvedAt")),
            m.get("resolution"),
            m.get("description") or m.get("resolutionSource"),
        )
        for m in markets
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO polymarket_markets
                (market_id, question, theme_tags, end_date, status,
                 created_at, resolved_at, resolved_outcome, resolution_text, fetched_at)
            VALUES %s
            ON CONFLICT (market_id) DO UPDATE SET
                question         = EXCLUDED.question,
                theme_tags       = EXCLUDED.theme_tags,
                end_date         = EXCLUDED.end_date,
                status           = EXCLUDED.status,
                resolved_at      = EXCLUDED.resolved_at,
                resolved_outcome = EXCLUDED.resolved_outcome,
                resolution_text  = EXCLUDED.resolution_text,
                fetched_at       = NOW()
            """,
            rows,
            template="(%s, %s, %s::text[], %s::date, %s, %s, %s, %s, %s, NOW())",
        )
        conn.commit()
    return len(rows)


def insert_price_snaps(conn, markets, snap_time):
    rows = []
    for m in markets:
        bid = parse_float(m.get("bestBid"))
        ask = parse_float(m.get("bestAsk"))
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
        rows.append((
            m["_market_id"],
            snap_time,
            bid,
            ask,
            mid,
            parse_float(m.get("lastTradePrice")),
            parse_float(m.get("volume24hr")),
        ))
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO polymarket_price_snap
                (market_id, time, bid, ask, mid, last, volume_24h)
            VALUES %s
            ON CONFLICT (market_id, time) DO NOTHING
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def upsert_position_snaps(conn, market_id, positions, snap_time):
    rows = []
    for pos in positions:
        side = (pos.get("side") or pos.get("outcome") or "UNKNOWN").upper()
        wallet = (
            pos.get("proxyWallet")
            or pos.get("userId")
            or pos.get("walletId")
            or ""
        )
        if not wallet:
            continue
        rows.append((
            market_id,
            snap_time,
            side,
            wallet,
            parse_float(pos.get("size") or pos.get("positionSize")),
            parse_float(pos.get("lifetimeWinRate") or pos.get("pnlPercentage")),
        ))
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO polymarket_position_snap
                (market_id, time, side, wallet_id, position_usd, wallet_lifetime_winrate)
            VALUES %s
            ON CONFLICT (market_id, time, side, wallet_id) DO NOTHING
            """,
            rows,
        )
        conn.commit()
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Truncate to current hour for idempotent hourly snapshots
    snap_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    print(f"{TAG} Starting — snapshot hour: {snap_time.isoformat()}")

    conn = psycopg2.connect(TSDB_DSN)
    positions_supported = True

    try:
        with conn.cursor() as cur:
            cur.execute(INIT_SQL)
            conn.commit()
        print(f"{TAG} Schema initialized")

        markets = fetch_all_theme_markets()
        if not markets:
            print(
                f"{TAG} WARNING: No theme-matched markets found — "
                "check keyword list or Gamma API availability",
                file=sys.stderr,
            )
            sys.exit(0)

        n_markets = upsert_markets(conn, markets)
        print(f"{TAG} Upserted {n_markets} market metadata rows")

        n_snaps = insert_price_snaps(conn, markets, snap_time)
        print(f"{TAG} Inserted {n_snaps} price snapshot rows (hour={snap_time.isoformat()})")

        # Position snapshots: best-effort, skip if endpoint requires auth
        n_pos_total = 0
        for m in markets:
            if not positions_supported:
                break
            positions = fetch_top_positions(m["_market_id"])
            if positions is None:
                print(f"{TAG} Position endpoint unavailable/requires auth — skipping position_snap")
                positions_supported = False
                break
            n_pos = upsert_position_snaps(conn, m["_market_id"], positions, snap_time)
            n_pos_total += n_pos
            time.sleep(REQUEST_DELAY)

        if positions_supported and n_pos_total > 0:
            print(f"{TAG} Inserted {n_pos_total} position rows across {len(markets)} markets")

        # Final stats
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM polymarket_markets WHERE status = 'active'")
            active_count = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM polymarket_price_snap WHERE time >= NOW() - INTERVAL '2 hours'"
            )
            recent_snaps = cur.fetchone()[0]

        print(
            f"{TAG} Done — {active_count} active markets tracked, "
            f"{recent_snaps} price snaps in last 2h"
        )

        if active_count < 20:
            print(
                f"{TAG} WARNING: only {active_count} active markets tracked — "
                "acceptance criteria requires ≥20; review keyword list",
                file=sys.stderr,
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()

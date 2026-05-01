-- QIDP-226 schema: FMP-driven sector/flow cache for crucix.
-- Idempotent — safe to re-run on existing databases.
--
-- Apply via:
--   psql "$QID_TSDB_DSN" -f fmp_schema.sql
-- or from inside the cluster:
--   kubectl -n qid exec -it deploy/qid-tsdb -- \
--     psql -U qid -d qid_analytics -f /tmp/fmp_schema.sql

-- Universe metadata: symbol → sector/industry/market_cap/exchange
-- One row per ticker. Refreshed weekly from FMP profile-bulk.
CREATE TABLE IF NOT EXISTS ticker_metadata (
    symbol         TEXT PRIMARY KEY,
    name           TEXT,
    sector         TEXT,
    industry       TEXT,
    market_cap     BIGINT,
    exchange       TEXT,
    is_etf         BOOLEAN,
    country        TEXT,
    last_refreshed TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ticker_metadata_sector_idx
    ON ticker_metadata (sector)
    WHERE country = 'US' AND NOT is_etf;

CREATE INDEX IF NOT EXISTS ticker_metadata_industry_idx
    ON ticker_metadata (industry)
    WHERE country = 'US' AND NOT is_etf;

-- Hourly market-movers snapshot. Hypertable so old rows roll off cheaply.
CREATE TABLE IF NOT EXISTS crucix_movers (
    time       TIMESTAMPTZ NOT NULL,
    category   TEXT NOT NULL,           -- 'gainers' | 'losers' | 'most_active'
    symbol     TEXT NOT NULL,
    name       TEXT,
    price      NUMERIC,
    change_pct NUMERIC,
    volume     BIGINT,
    exchange   TEXT,
    PRIMARY KEY (time, category, symbol)
);

SELECT create_hypertable(
    'crucix_movers', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days'
);

CREATE INDEX IF NOT EXISTS crucix_movers_symbol_time_idx
    ON crucix_movers (symbol, time DESC);

-- Daily sector breadth (avg %change) by exchange.
CREATE TABLE IF NOT EXISTS crucix_sector_snap (
    time       TIMESTAMPTZ NOT NULL,
    sector     TEXT NOT NULL,
    exchange   TEXT NOT NULL,
    avg_change NUMERIC,
    PRIMARY KEY (time, sector, exchange)
);

SELECT create_hypertable(
    'crucix_sector_snap', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days'
);

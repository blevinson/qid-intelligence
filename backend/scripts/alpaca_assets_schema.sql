-- Alpaca tradable universe cache.
-- One row per Alpaca asset; refreshed daily from /v2/assets.
-- Joined to ticker_metadata.symbol for sector/industry enrichment.
CREATE TABLE IF NOT EXISTS alpaca_assets (
    symbol           TEXT PRIMARY KEY,
    name             TEXT,
    exchange         TEXT,
    asset_class      TEXT,
    status           TEXT,
    tradable         BOOLEAN,
    marginable       BOOLEAN,
    shortable        BOOLEAN,
    easy_to_borrow   BOOLEAN,
    fractionable     BOOLEAN,
    attributes       TEXT[],
    last_refreshed   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS alpaca_assets_exchange_idx
    ON alpaca_assets (exchange) WHERE tradable;

-- Convenience view: Alpaca tradable × FMP sector/industry.
-- Symbol normalisation handles BRK.B/BRK-B style mismatches.
CREATE OR REPLACE VIEW alpaca_x_industry AS
SELECT
    a.symbol            AS alpaca_symbol,
    COALESCE(m.symbol, m2.symbol) AS fmp_symbol,
    a.name,
    a.exchange          AS exchange_alpaca,
    COALESCE(m.exchange, m2.exchange) AS exchange_fmp,
    COALESCE(m.sector, m2.sector)     AS sector,
    COALESCE(m.industry, m2.industry) AS industry,
    COALESCE(m.is_etf, m2.is_etf)     AS is_etf,
    a.tradable,
    a.attributes
FROM alpaca_assets a
LEFT JOIN ticker_metadata m  ON m.symbol  = a.symbol
LEFT JOIN ticker_metadata m2 ON m2.symbol = REPLACE(a.symbol, '.', '-')
WHERE a.tradable AND a.asset_class = 'us_equity';

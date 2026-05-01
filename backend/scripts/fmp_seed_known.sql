-- Pre-seed ticker_metadata with the names Sonnet actually generates ideas for,
-- so Layers B+C of QIDP-226 can work immediately while the FMP bulk endpoint
-- backfills the long tail (parts 2-3 still rate-limited).
--
-- Sectors normalized to FMP's vocabulary (Communication Services, Consumer
-- Cyclical, Consumer Defensive, Financial Services, etc.) so when the bulk
-- endpoint UPSERTs over these rows, the sector column doesn't churn.
--
-- ON CONFLICT DO UPDATE — keep these authoritative until FMP overwrites them
-- via a successful profile-bulk run. NULL fields stay NULL.

BEGIN;

INSERT INTO ticker_metadata (symbol, name, sector, industry, exchange, is_etf, country, last_refreshed) VALUES
  -- Mega-cap stocks (yfinance SECTOR_BY_SYMBOL)
  ('AAPL',  'Apple Inc.',                     'Technology',             'Consumer Electronics',     'NASDAQ', false, 'US', NOW()),
  ('MSFT',  'Microsoft Corp.',                'Technology',             'Software Infrastructure',  'NASDAQ', false, 'US', NOW()),
  ('NVDA',  'NVIDIA Corp.',                   'Technology',             'Semiconductors',           'NASDAQ', false, 'US', NOW()),
  ('GOOGL', 'Alphabet Inc.',                  'Communication Services', 'Internet Content',         'NASDAQ', false, 'US', NOW()),
  ('META',  'Meta Platforms Inc.',            'Communication Services', 'Internet Content',         'NASDAQ', false, 'US', NOW()),
  ('AMZN',  'Amazon.com Inc.',                'Consumer Cyclical',      'Internet Retail',          'NASDAQ', false, 'US', NOW()),
  ('TSLA',  'Tesla Inc.',                     'Consumer Cyclical',      'Auto Manufacturers',       'NASDAQ', false, 'US', NOW()),
  ('XOM',   'Exxon Mobil Corp.',              'Energy',                 'Oil & Gas Integrated',     'NYSE',   false, 'US', NOW()),
  ('CVX',   'Chevron Corp.',                  'Energy',                 'Oil & Gas Integrated',     'NYSE',   false, 'US', NOW()),
  ('JPM',   'JPMorgan Chase & Co.',           'Financial Services',     'Banks - Diversified',      'NYSE',   false, 'US', NOW()),
  ('V',     'Visa Inc.',                      'Financial Services',     'Credit Services',          'NYSE',   false, 'US', NOW()),
  ('UNH',   'UnitedHealth Group Inc.',        'Healthcare',             'Healthcare Plans',         'NYSE',   false, 'US', NOW()),
  ('WMT',   'Walmart Inc.',                   'Consumer Defensive',     'Discount Stores',          'NYSE',   false, 'US', NOW()),

  -- Defense names that come up in geopolitical-narrative ideas
  ('RTX',   'RTX Corporation',                'Industrials',            'Aerospace & Defense',      'NYSE',   false, 'US', NOW()),
  ('NOC',   'Northrop Grumman Corp.',         'Industrials',            'Aerospace & Defense',      'NYSE',   false, 'US', NOW()),
  ('LMT',   'Lockheed Martin Corp.',          'Industrials',            'Aerospace & Defense',      'NYSE',   false, 'US', NOW()),
  ('BA',    'Boeing Co.',                     'Industrials',            'Aerospace & Defense',      'NYSE',   false, 'US', NOW()),
  ('GD',    'General Dynamics Corp.',         'Industrials',            'Aerospace & Defense',      'NYSE',   false, 'US', NOW()),
  ('BWXT',  'BWX Technologies Inc.',          'Industrials',            'Aerospace & Defense',      'NYSE',   false, 'US', NOW()),

  -- Energy / commodity names recently mentioned
  ('STNG',  'Scorpio Tankers Inc.',           'Energy',                 'Oil & Gas Midstream',      'NYSE',   false, 'US', NOW()),
  ('OXY',   'Occidental Petroleum Corp.',     'Energy',                 'Oil & Gas E&P',            'NYSE',   false, 'US', NOW()),
  ('SLB',   'Schlumberger Ltd.',              'Energy',                 'Oil & Gas Equipment',      'NYSE',   false, 'US', NOW()),

  -- Sector ETFs — overridden to actual sector (FMP would tag "Financial Services / Asset Management")
  ('XLE',   'Energy Select Sector SPDR',         'Energy',                 'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLB',   'Materials Select Sector SPDR',      'Basic Materials',        'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLI',   'Industrials Select Sector SPDR',    'Industrials',            'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLY',   'Consumer Discretionary Select Sector SPDR', 'Consumer Cyclical', 'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLP',   'Consumer Staples Select Sector SPDR','Consumer Defensive',    'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLV',   'Health Care Select Sector SPDR',    'Healthcare',             'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLF',   'Financial Select Sector SPDR',      'Financial Services',     'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLK',   'Technology Select Sector SPDR',     'Technology',             'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLC',   'Communication Services Select Sector SPDR', 'Communication Services', 'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLU',   'Utilities Select Sector SPDR',      'Utilities',              'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('XLRE',  'Real Estate Select Sector SPDR',    'Real Estate',            'ETF', 'NYSEARCA', true, 'US', NOW()),

  -- Theme ETFs that appear in ideas
  ('ITA',   'iShares U.S. Aerospace & Defense ETF',       'Industrials', 'ETF', 'CBOE',     true, 'US', NOW()),
  ('URA',   'Global X Uranium ETF',                       'Energy',      'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('USO',   'United States Oil Fund LP',                  'Energy',      'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('UCO',   'ProShares Ultra Bloomberg Crude Oil',        'Energy',      'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('BNO',   'United States Brent Oil Fund',               'Energy',      'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('UNG',   'United States Natural Gas Fund',             'Energy',      'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('GLD',   'SPDR Gold Shares',                           'Basic Materials', 'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('SLV',   'iShares Silver Trust',                       'Basic Materials', 'ETF', 'NYSEARCA', true, 'US', NOW()),

  -- Vol / hedging products — no clean GICS sector, leave NULL
  ('UVXY',  'ProShares Ultra VIX Short-Term Futures',     NULL, 'ETF', 'CBOE', true, 'US', NOW()),
  ('VXX',   'iPath Series B S&P 500 VIX Short-Term',      NULL, 'ETF', 'CBOE', true, 'US', NOW()),
  ('TLT',   'iShares 20+ Year Treasury Bond ETF',         NULL, 'ETF', 'NASDAQ', true, 'US', NOW()),
  ('HYG',   'iShares iBoxx High Yield Corporate Bond',    NULL, 'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('LQD',   'iShares iBoxx Investment Grade Corporate',   NULL, 'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('SPY',   'SPDR S&P 500 ETF Trust',                     NULL, 'ETF', 'NYSEARCA', true, 'US', NOW()),
  ('QQQ',   'Invesco QQQ Trust',                          'Technology', 'ETF', 'NASDAQ', true, 'US', NOW())
ON CONFLICT (symbol) DO UPDATE SET
  name = COALESCE(EXCLUDED.name, ticker_metadata.name),
  sector = COALESCE(EXCLUDED.sector, ticker_metadata.sector),
  industry = COALESCE(EXCLUDED.industry, ticker_metadata.industry),
  exchange = COALESCE(EXCLUDED.exchange, ticker_metadata.exchange),
  is_etf = EXCLUDED.is_etf,
  country = COALESCE(EXCLUDED.country, ticker_metadata.country),
  last_refreshed = NOW();

COMMIT;

-- Sanity check
SELECT sector, count(*)
FROM ticker_metadata
WHERE symbol IN (
  'AAPL','MSFT','NVDA','GOOGL','META','AMZN','TSLA','XOM','CVX','JPM','V','UNH','WMT',
  'RTX','NOC','LMT','BA','GD','BWXT','STNG','OXY','SLB',
  'XLE','XLB','XLI','XLY','XLP','XLV','XLF','XLK','XLC','XLU','XLRE',
  'ITA','URA','USO','UCO','BNO','UNG','GLD','SLV',
  'UVXY','VXX','TLT','HYG','LQD','SPY','QQQ'
)
GROUP BY sector
ORDER BY count DESC;

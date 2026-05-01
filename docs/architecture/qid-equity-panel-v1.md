# qid-equity panel v1

Materialized daily feature panel for backtesting and paper-period decisions
across the breakout / mean-reversion / rotation / event lanes. Built once,
read everywhere — DuckDB / Polars over MinIO parquet, sub-second iteration.

## Layout

```
s3://qid-equity/
├── bars/ohlcv-1d/{symbol}.parquet     # raw daily OHLCV (Alpaca SIP)
├── universe/snapshot.json              # daily breakout universe
└── panel/v1/dt=YYYY-MM-DD/panel.parquet
```

`v1/` is the schema lock; future schema breaks land at `v2/` so consumers can
pin.

## Pipelines

| Job                       | Schedule (UTC)        | Output                                              |
| ------------------------- | --------------------- | --------------------------------------------------- |
| `alpaca-bars-ingest`      | 22:12, weekdays       | append `bars/ohlcv-1d/{symbol}.parquet`             |
| `universe-export`         | 22:30, weekdays       | overwrite `universe/snapshot.json`                  |
| `panel-build` (daily)     | 22:45, weekdays       | last 30 partitions of `panel/v1/dt=*/panel.parquet` |
| `panel-build-cold` (Job)  | manual                | full-history rebuild of `panel/v1/`                 |

All three read MinIO secret `minio-credentials`; bars + panel hit Alpaca SIP
(no IEX fallback — fail-loud on non-200). Bar ingest is idempotent (appends
from `max(dt) + 1`); panel cold rebuild is destructive within `panel/v1/`.

## Universe

`universe/snapshot.json` schema:

```json
{
  "schema_version": 1,
  "generated_at_utc": "2026-05-01T22:30:00+00:00",
  "filter": {
    "min_market_cap_usd": 2.0e9,
    "exchanges": ["NASDAQ","NYSE","AMEX","ARCA"],
    "etf_allowlist": ["SPY","QQQ","XLE","XLF",...]
  },
  "count": 2287,
  "symbols": [{ "symbol": "AAPL", "sector": "...", ... }, ...]
}
```

Filter: `tradable AND asset_class='us_equity' AND exchange IN
(NASDAQ,NYSE,AMEX,ARCA) AND (market_cap >= $2B OR symbol IN etf_allowlist)`.
ETF allowlist is hand-curated — 11 GICS sector SPDRs + theme/catalyst ETFs
(ITA, USO, GLD, TLT, HYG, SMH, ARKK, FXI, etc.) — so niche themes survive
even when their cap drifts under $2B.

## Panel schema (v1) — 53 columns

| Group                | Columns                                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Identity             | `dt`, `symbol`, `sector`, `industry`, `market_cap`, `is_etf`                                                                     |
| OHLCV                | `open`, `high`, `low`, `close`, `volume`, `trade_count`, `vwap`                                                                  |
| Returns              | `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`                                                                                         |
| Vol / momentum       | `realized_vol_20d`, `realized_vol_60d`, `realized_vol_60d_pctile`, `atr14`, `atr14_normalized`, `rsi_14`, `mom_z_5_60`            |
| Trend distance       | `distance_from_20ma_pct`, `distance_from_50ma_pct`, `z_score_close_vs_20ma`, `z_score_close_vs_50ma`                              |
| Breakout markers     | `pct_to_20d_high`, `pct_to_60d_high`, `high20_dist`, `low20_dist`                                                                |
| Volume               | `dollar_volume`, `dollar_volume_20d_avg`, `volume_ratio_today`, `volume_z_20`                                                    |
| Microstructure       | `gap_open`, `range_pos`, `vwap20_ratio`                                                                                          |
| Forward labels (ML)  | `fwd_return_1d`, `fwd_return_5d`, `fwd_return_10d`, `fwd_return_20d`, `fwd_max_runup_5d`, `fwd_max_drawdown_5d`                  |
| Crucix overlay (NaN) | `n_ideas_24h`, `n_ideas_7d`, `theme_long_count_industry_7d`, `theme_short_count_industry_7d`, `industry_flow_rank`, `sector_flow_rank`, `earnings_date_proximity_days`, `days_since_corporate_action` |

### No-leak invariants

- All rolling-window features use **trailing** windows ending at `dt` (close).
- Forward-return labels use bars **strictly after** `dt`. They are NaN on the
  current trailing edge — readers should drop NaN labels before training.
- Crucix overlay columns are reserved as NaN today. Backfill is intentionally
  not attempted (crucix history is weeks, not years; no point-in-time
  reconstruction). Tier-B/C consumers fill these forward from the live
  crucix path; backtests treat them as missing.

## Reading the panel

DuckDB over httpfs (1y sweep is sub-second cold, ~50ms warm):

```python
import duckdb
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("""
    SET s3_endpoint='minio.qid.svc.cluster.local:9000';
    SET s3_url_style='path';
    SET s3_use_ssl=false;
    SET s3_access_key_id=getenv('MINIO_ACCESS_KEY');
    SET s3_secret_access_key=getenv('MINIO_SECRET_KEY');
""")
df = con.execute("""
    SELECT *
    FROM read_parquet('s3://qid-equity/panel/v1/dt=2025-*/panel.parquet',
                      hive_partitioning=true)
    WHERE symbol = 'NVDA'
""").df()
```

Polars / pandas via `pyarrow.fs.S3FileSystem` works identically.

## Verification (v1 baseline, 2026-05-01)

- 1,256 partitions / 2,664,017 rows / 0 failures (cold rebuild)
- 1y DuckDB fill-rate sweep over 250 dates / 553,178 rows: every bar-derived
  column ≥ 99.6% filled; crucix overlay 100% NaN as designed.

## Known limitations

- **Survivorship bias** — universe is today's mid+large names projected back
  5y. Historical returns are biased upward. Point-in-time universe is a
  later upgrade.
- **Cash dividends / splits** — `adjustment=raw` from Alpaca. Total-return
  alignment lives in feature engineering, not in the bar layer.
- **ETF flow features** — `*_flow_rank` is a crucix-derived placeholder.
  Tier-B work fills it; backtests must tolerate NaN there.

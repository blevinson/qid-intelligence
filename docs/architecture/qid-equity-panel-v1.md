# qid-equity panel v1 — schema reference

**Version**: v1.1 (QIDP-231 — added 4 consolidation-gate columns)
**Storage**: `s3://qid-equity/panel/v1/dt=YYYY-MM-DD/panel.parquet` (hive-partitioned)
**Producer**: `qid-intelligence/backend/scripts/panel_build.py`
**Consumer**: `tradefarm/src/tradefarm/strategies/panel_loader.py`

---

## Column reference (57 total)

### Identity (6)

| Column | Type | Description |
|---|---|---|
| `dt` | date | Trading date (partition key) |
| `symbol` | str | Ticker (uppercase, US equity) |
| `sector` | str | GICS sector |
| `industry` | str | GICS industry group |
| `market_cap` | int | Latest market cap (USD) |
| `is_etf` | bool | True if the ticker is an ETF |

### OHLCV (7)

| Column | Type | Description |
|---|---|---|
| `open` | float | Adjusted open |
| `high` | float | Adjusted high |
| `low` | float | Adjusted low |
| `close` | float | Adjusted close |
| `volume` | float | Share volume |
| `trade_count` | float | Number of trades (Databento; NaN for non-Databento sources) |
| `vwap` | float | Volume-weighted average price (Databento; NaN otherwise) |

### Returns (4)

| Column | Type | Description |
|---|---|---|
| `ret_1d` | float | 1-day return (close[t] / close[t-1] − 1) |
| `ret_5d` | float | 5-day return |
| `ret_10d` | float | 10-day return |
| `ret_20d` | float | 20-day return |

### Volatility / momentum (7)

| Column | Type | Description |
|---|---|---|
| `realized_vol_20d` | float | Annualized realized vol, 20-day log-return window |
| `realized_vol_60d` | float | Annualized realized vol, 60-day window |
| `realized_vol_60d_pctile` | float | Cross-sectional percentile rank of `realized_vol_60d` (0–1) |
| `atr14` | float | ATR-14 (EWM, True Range definition) |
| `atr14_normalized` | float | `atr14 / close` clipped to [0, 1] |
| `rsi_14` | float | RSI-14 (EWM, standard Wilder smoothing) |
| `mom_z_5_60` | float | Z-score: 5d return normalized by rolling 60d distribution of 5d returns |

### Trend distance (4)

| Column | Type | Description |
|---|---|---|
| `distance_from_20ma_pct` | float | `(close − SMA20) / SMA20` |
| `distance_from_50ma_pct` | float | `(close − SMA50) / SMA50` |
| `z_score_close_vs_20ma` | float | `(close − SMA20) / rolling_std20` |
| `z_score_close_vs_50ma` | float | `(close − SMA50) / rolling_std50` |

### Breakout markers (4)

Sign conventions:

- `pct_to_60d_high = (high60_excl − close) / close` — **negative = close at/above 60d high**.
  A "fresh 60d high" filter is `pct_to_60d_high <= 0`.
- `high20_dist = (close − high20_excl) / high20_excl` — **positive = close above prior 20d high**.

| Column | Type | Description |
|---|---|---|
| `pct_to_20d_high` | float | `(prior_20d_high − close) / close` (ex-today rolling) |
| `pct_to_60d_high` | float | `(prior_60d_high − close) / close` (ex-today rolling) |
| `high20_dist` | float | `(close − prior_20d_high) / prior_20d_high` |
| `low20_dist` | float | `(close − prior_20d_low) / prior_20d_low` |

### Volume (4)

| Column | Type | Description |
|---|---|---|
| `dollar_volume` | float | `close × volume` |
| `dollar_volume_20d_avg` | float | 20-day rolling mean of `dollar_volume` |
| `volume_ratio_today` | float | `volume / rolling_20d_mean(volume)` |
| `volume_z_20` | float | Z-score of volume vs 20-day distribution |

### Microstructure (3)

| Column | Type | Description |
|---|---|---|
| `gap_open` | float | `(open − prev_close) / prev_close` |
| `range_pos` | float | `(close − low) / (high − low)` — 0 = day-low close, 1 = day-high |
| `vwap20_ratio` | float | `vwap / rolling_20d_mean(vwap)` |

### Forward labels — ML only (6)

The harness strips all `fwd_*` columns before handing the slice to a Strategy.
A strategy cannot see these. They are NaN on the trailing edge by design.

| Column | Type | Description |
|---|---|---|
| `fwd_return_1d` | float | Next-day return |
| `fwd_return_5d` | float | 5-day cumulative return |
| `fwd_return_10d` | float | 10-day cumulative return |
| `fwd_return_20d` | float | 20-day cumulative return |
| `fwd_max_runup_5d` | float | Max intraday runup over next 5 days |
| `fwd_max_drawdown_5d` | float | Max intraday drawdown over next 5 days |

### Crucix overlay — reserved NaN today (8)

These columns are present but fully NaN until the crucix overlay pipeline
populates them. Do **not** block a strategy on them — use `notna().any()` to
detect populated state first.

| Column | Type | Description |
|---|---|---|
| `n_ideas_24h` | float | Crucix idea count in last 24h for this symbol |
| `n_ideas_7d` | float | Crucix idea count in last 7 days |
| `theme_long_count_industry_7d` | float | Long-theme count, industry, 7d |
| `theme_short_count_industry_7d` | float | Short-theme count, industry, 7d |
| `industry_flow_rank` | float | Crucix industry-level flow rank (0–1) |
| `sector_flow_rank` | float | Crucix sector-level flow rank (0–1) |
| `earnings_date_proximity_days` | float | Calendar days to nearest earnings date |
| `days_since_corporate_action` | float | Days since last split/dividend/restatement |

### Consolidation-gate columns — v1.1 (QIDP-231) (4)

Added by FIN-1145. Required by TFARM-40 to replace the v0 flat-base gate.
Without these columns the breakout strategy runs in all-pass fallback mode.

| Column | Type | Formula | Gate |
|---|---|---|---|
| `bb_squeeze_ratio` | float | `(4·std20/SMA20) / (4·ATR14/EMA20)` = `(std20·EMA20) / (SMA20·ATR14)` — **< 1.0 → BB inside KC (squeeze)** | `bb_squeeze` |
| `atr14_normalized_lag20` | float | `atr14_normalized` shifted 20 trading days back per symbol | `declining_atr` |
| `range20_norm` | float | `(rolling_20d_high − rolling_20d_low) / close` (inclusive 20-day window) | `range_compression` |
| `close_dev30_norm` | float | `abs(close − SMA30) / close` | `sideways` |

All 4 are strictly trailing (no look-ahead). They have `NaN` for the first
20–30 rows per symbol (warm-up period).

---

## No-leak invariants

1. All rolling windows at row `t` use only data ≤ `t`.
2. `pct_to_20d_high` and `pct_to_60d_high` use `high.shift(1).rolling(N).max()`
   (prior N days, excluding today).
3. Forward-return labels use `close.pct_change(N).shift(-N)` — they are NaN
   on the trailing N rows by design.
4. Cross-sectional features (e.g. `realized_vol_60d_pctile`) rank within the
   same calendar date only — no time contamination.

---

## Data pipeline

```
Databento XNAS.BASIC (daily bars)
    → MinIO s3://qid-equity/bars/{SYMBOL}.parquet   (per-symbol OHLCV)
    → panel_build.py                                (QIDP compute + write)
    → MinIO s3://qid-equity/panel/v1/dt=*/panel.parquet
    → tradefarm panel_loader.py (DuckDB httpfs read)
    → Strategy callable (Strategy.py)
```

```
TimescaleDB qid_analytics.ticker_metadata
    → panel_build.py                                (sector / industry / market_cap / is_etf join)
```

---

## Running panel_build.py

```bash
# Cold rebuild (5y)
python backend/scripts/panel_build.py --cold-rebuild

# Daily incremental (yesterday)
python backend/scripts/panel_build.py

# Specific range
python backend/scripts/panel_build.py --start 2024-01-01 --end 2024-12-31

# Validate fill rate (DuckDB sweep, exits 0 if all >=95%)
python backend/scripts/panel_build.py --validate
```

Required env: `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`,
`QID_DB_HOST`, `QID_DB_PORT`, `QID_DB_NAME`, `QID_DB_USER`, `QID_DB_PASS`.

---

## Change history

| Version | Ticket | Change |
|---|---|---|
| v1.0 | TFARM-28, TFARM-31 | Initial 53-column schema |
| v1.1 | QIDP-231 / FIN-1145 | Added 4 consolidation-gate columns (`bb_squeeze_ratio`, `atr14_normalized_lag20`, `range20_norm`, `close_dev30_norm`) |

#!/usr/bin/env python3
"""qid-equity panel v1.1 builder — FIN-1145 / QIDP-231.

Reads per-symbol OHLCV bars from s3://qid-equity/bars/{SYMBOL}.parquet,
computes all panel v1.1 columns (57 total: 53 existing + 4 new consolidation-
gate columns from QIDP-231), and writes daily hive-partitioned parquet to
s3://qid-equity/panel/v1/dt=YYYY-MM-DD/panel.parquet.

Usage:
    # Cold rebuild — full 5-year history:
    python panel_build.py --cold-rebuild

    # Daily incremental (yesterday's partition, default):
    python panel_build.py

    # Explicit date range:
    python panel_build.py --start 2024-01-01 --end 2024-12-31

    # DuckDB fill-rate validation:
    python panel_build.py --validate

Env:
    MINIO_ENDPOINT       default: minio.qid.svc.cluster.local:9000
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    MINIO_SECURE         default: false
    QID_EQUITY_BUCKET    default: qid-equity
    QID_DB_HOST          default: qid-tsdb-rw.qid.svc.cluster.local
    QID_DB_PORT          default: 5432
    QID_DB_NAME          default: qid_analytics
    QID_DB_USER
    QID_DB_PASS

New columns (QIDP-231 / FIN-1145):
    bb_squeeze_ratio       — BB width / KC width (ratio < 1 → squeeze)
    atr14_normalized_lag20 — atr14_normalized shifted 20 trading days back
    range20_norm           — (rolling_20d_high − rolling_20d_low) / close
    close_dev30_norm       — abs(close − sma30) / close
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from datetime import date, timedelta
from typing import Generator

import duckdb
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from minio import Minio
from minio.error import S3Error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio.qid.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
QID_EQUITY_BUCKET = os.environ.get("QID_EQUITY_BUCKET", "qid-equity")

TSDB_DSN = os.environ.get("QID_TSDB_DSN", (
    f"host={os.environ.get('QID_DB_HOST', 'qid-tsdb-rw.qid.svc.cluster.local')} "
    f"port={os.environ.get('QID_DB_PORT', '5432')} "
    f"dbname={os.environ.get('QID_DB_NAME', 'qid_analytics')} "
    f"user={os.environ.get('QID_DB_USER', '')} "
    f"password={os.environ.get('QID_DB_PASS', '')}"
))

BARS_PREFIX = "bars"
PANEL_PREFIX = "panel/v1"
UNIVERSE_KEY = "universe/snapshot.json"

# Cold rebuild lookback: ~5 trading years
COLD_REBUILD_START = date(2021, 5, 3)

# Minimum history required before a feature row is considered valid.
_MIN_HISTORY_DAYS = 60

log = logging.getLogger("panel_build")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def _minio_client() -> Minio:
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError("MINIO_ACCESS_KEY / MINIO_SECRET_KEY not set")
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def _read_parquet_from_minio(client: Minio, key: str) -> pd.DataFrame | None:
    try:
        resp = client.get_object(QID_EQUITY_BUCKET, key)
        try:
            buf = io.BytesIO(resp.read())
        finally:
            resp.close()
            resp.release_conn()
        return pd.read_parquet(buf)
    except S3Error as e:
        if e.code == "NoSuchKey":
            return None
        raise


def _write_parquet_to_minio(client: Minio, key: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    size = buf.tell()
    buf.seek(0)
    client.put_object(
        QID_EQUITY_BUCKET,
        key,
        buf,
        length=size,
        content_type="application/x-parquet",
    )


def _list_bar_symbols(client: Minio) -> list[str]:
    """Return all symbol names present in bars/ prefix."""
    symbols = []
    for obj in client.list_objects(QID_EQUITY_BUCKET, prefix=f"{BARS_PREFIX}/"):
        name = obj.object_name
        if name.endswith(".parquet"):
            sym = name[len(BARS_PREFIX) + 1 : -len(".parquet")].upper()
            symbols.append(sym)
    return sorted(symbols)


# ---------------------------------------------------------------------------
# Universe metadata
# ---------------------------------------------------------------------------

def _load_universe_metadata(conn) -> pd.DataFrame:
    """Load symbol metadata from ticker_metadata.

    Returns DataFrame with columns: symbol, sector, industry, market_cap, is_etf.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                symbol,
                sector,
                industry,
                market_cap,
                is_etf
            FROM ticker_metadata
        """)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "sector", "industry", "market_cap", "is_etf"]
        )
    return pd.DataFrame([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Per-symbol feature computation
# ---------------------------------------------------------------------------

_EPS = 1e-9


def _compute_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-symbol time-series features from raw OHLCV bars.

    Input columns expected: date, open, high, low, close, volume,
    trade_count (optional), vwap (optional).

    Returns a DataFrame with one row per date for this symbol.
    All features are strictly trailing (no leak).
    """
    df = df.sort_values("date").reset_index(drop=True).copy()

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["dt"] = pd.to_datetime(df["date"]).dt.date
    out["open"] = open_
    out["high"] = high
    out["low"] = low
    out["close"] = close
    out["volume"] = volume
    out["trade_count"] = df.get("trade_count", pd.Series(np.nan, index=df.index))
    out["vwap"] = df.get("vwap", pd.Series(np.nan, index=df.index))

    # Returns
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)

    # ATR-14
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    out["atr14"] = atr14
    out["atr14_normalized"] = (atr14 / close.replace(0, np.nan)).clip(0, 1.0)

    # RSI-14
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Realized vol
    log_ret = np.log(close / close.shift(1).replace(0, np.nan))
    out["realized_vol_20d"] = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    out["realized_vol_60d"] = log_ret.rolling(60, min_periods=30).std() * np.sqrt(252)

    # Moving averages
    sma20 = close.rolling(20, min_periods=10).mean()
    sma30 = close.rolling(30, min_periods=15).mean()
    sma50 = close.rolling(50, min_periods=25).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    out["distance_from_20ma_pct"] = (close - sma20) / sma20.replace(0, np.nan)
    out["distance_from_50ma_pct"] = (close - sma50) / sma50.replace(0, np.nan)

    std20 = close.rolling(20, min_periods=10).std()
    out["z_score_close_vs_20ma"] = (close - sma20) / std20.replace(0, np.nan)
    std50 = close.rolling(50, min_periods=25).std()
    out["z_score_close_vs_50ma"] = (close - sma50) / std50.replace(0, np.nan)

    # Momentum z-score: 5d vs 60d rolling return distributions
    ret_5d = close.pct_change(5)
    ret_60d_mean = ret_5d.rolling(60, min_periods=20).mean()
    ret_60d_std = ret_5d.rolling(60, min_periods=20).std()
    out["mom_z_5_60"] = (ret_5d - ret_60d_mean) / ret_60d_std.replace(0, np.nan)

    # Breakout markers: excluding current day (strict trailing)
    # high20_excl = max of high over the previous 20 days (not including today)
    high20_excl = high.shift(1).rolling(20, min_periods=10).max()
    low20_excl = low.shift(1).rolling(20, min_periods=10).min()
    high60_excl = high.shift(1).rolling(60, min_periods=30).max()

    out["pct_to_20d_high"] = (high20_excl - close) / close.replace(0, np.nan)
    out["pct_to_60d_high"] = (high60_excl - close) / close.replace(0, np.nan)
    # positive = close above prior 20d high
    out["high20_dist"] = (close - high20_excl) / high20_excl.replace(0, np.nan)
    out["low20_dist"] = (close - low20_excl) / low20_excl.replace(0, np.nan)

    # Volume features
    dollar_vol = close * volume
    dollar_vol_20d_avg = dollar_vol.rolling(20, min_periods=10).mean()
    vol_mean20 = volume.rolling(20, min_periods=10).mean()
    vol_std20 = volume.rolling(20, min_periods=10).std().replace(0, np.nan)

    out["dollar_volume"] = dollar_vol
    out["dollar_volume_20d_avg"] = dollar_vol_20d_avg
    out["volume_ratio_today"] = volume / vol_mean20.replace(0, np.nan)
    out["volume_z_20"] = (volume - vol_mean20) / vol_std20

    # Microstructure
    prev_close_shifted = close.shift(1).replace(0, np.nan)
    out["gap_open"] = (open_ - prev_close_shifted) / prev_close_shifted
    out["range_pos"] = (close - low) / (high - low).replace(0, np.nan)

    vwap_series = df.get("vwap", pd.Series(np.nan, index=df.index)).astype(float)
    vwap20_mean = vwap_series.rolling(20, min_periods=10).mean()
    out["vwap20_ratio"] = vwap_series / vwap20_mean.replace(0, np.nan)

    # Forward return labels (NaN at trailing edge — harness uses these)
    out["fwd_return_1d"] = close.pct_change(1).shift(-1)
    out["fwd_return_5d"] = close.pct_change(5).shift(-5)
    out["fwd_return_10d"] = close.pct_change(10).shift(-10)
    out["fwd_return_20d"] = close.pct_change(20).shift(-20)

    # Forward max runup / drawdown (5d window)
    def _fwd_runup(c: pd.Series, horizon: int) -> pd.Series:
        result = pd.Series(np.nan, index=c.index)
        for i in range(len(c)):
            window = c.iloc[i + 1 : i + 1 + horizon]
            if len(window) == horizon:
                result.iloc[i] = (window.max() - c.iloc[i]) / c.iloc[i]
        return result

    def _fwd_drawdown(c: pd.Series, horizon: int) -> pd.Series:
        result = pd.Series(np.nan, index=c.index)
        for i in range(len(c)):
            window = c.iloc[i + 1 : i + 1 + horizon]
            if len(window) == horizon:
                result.iloc[i] = (window.min() - c.iloc[i]) / c.iloc[i]
        return result

    out["fwd_max_runup_5d"] = _fwd_runup(close, 5)
    out["fwd_max_drawdown_5d"] = _fwd_drawdown(close, 5)

    # Crucix overlay — reserved as NaN today
    for col in [
        "n_ideas_24h",
        "n_ideas_7d",
        "theme_long_count_industry_7d",
        "theme_short_count_industry_7d",
        "industry_flow_rank",
        "sector_flow_rank",
        "earnings_date_proximity_days",
        "days_since_corporate_action",
    ]:
        out[col] = np.nan

    # -----------------------------------------------------------------------
    # QIDP-231 / FIN-1145: 4 new consolidation-gate columns
    # -----------------------------------------------------------------------

    # 1. bb_squeeze_ratio = (BB_width / SMA20) / (KC_width / EMA20)
    #    BB_width  = (upper_bb - lower_bb) = 4 * std20
    #    KC_width  = (upper_kc - lower_kc) = 4 * ATR14
    #    ratio < 1.0 → Bollinger Bands fit inside Keltner Channel (squeeze)
    bb_width = 4.0 * std20
    kc_width = 4.0 * atr14
    out["bb_squeeze_ratio"] = (bb_width / sma20.replace(0, np.nan)) / (
        kc_width / ema20.replace(0, np.nan)
    )

    # 2. atr14_normalized_lag20 = atr14_normalized shifted 20 trading days back
    #    Measures whether ATR is contracting vs 20 days ago.
    out["atr14_normalized_lag20"] = out["atr14_normalized"].shift(20)

    # 3. range20_norm = (rolling_20d_high - rolling_20d_low) / close
    #    Uses inclusive rolling (today + prior 19), normalized by close.
    high20_incl = high.rolling(20, min_periods=10).max()
    low20_incl = low.rolling(20, min_periods=10).min()
    out["range20_norm"] = (high20_incl - low20_incl) / close.replace(0, np.nan)

    # 4. close_dev30_norm = abs(close - sma_30d) / close
    out["close_dev30_norm"] = (close - sma30).abs() / close.replace(0, np.nan)

    return out


# ---------------------------------------------------------------------------
# Cross-sectional feature finalization
# ---------------------------------------------------------------------------

def _add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute features that require the full cross-section for a given date.

    Currently: realized_vol_60d_pctile (rank within the daily universe).
    """
    panel = panel.copy()
    vol60 = panel["realized_vol_60d"]
    panel["realized_vol_60d_pctile"] = vol60.rank(pct=True, na_option="keep")
    return panel


# ---------------------------------------------------------------------------
# Panel assembly pipeline
# ---------------------------------------------------------------------------

def _iter_trading_dates(
    start: date, end: date, all_dates: set[date]
) -> Generator[date, None, None]:
    """Yield sorted trading dates in [start, end] that exist in all_dates."""
    for d in sorted(all_dates):
        if start <= d <= end:
            yield d


def _load_all_bars(
    client: Minio,
    symbols: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV bars for all symbols into memory, filtered to [start, end].

    Returns {symbol: df} where df has columns date, open, high, low, close, etc.
    Symbols missing from MinIO are silently skipped.
    """
    # Load a bit before start to warm rolling windows
    load_start = start - timedelta(days=120)
    symbol_bars: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        key = f"{BARS_PREFIX}/{sym}.parquet"
        df = _read_parquet_from_minio(client, key)
        if df is None or df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= load_start) & (df["date"] <= end)].copy()
        if df.empty:
            continue
        symbol_bars[sym] = df
        if (i + 1) % 200 == 0:
            log.info("Loaded bars for %d / %d symbols", i + 1, len(symbols))
    log.info("Loaded bars for %d symbols total", len(symbol_bars))
    return symbol_bars


def build_panel(
    start: date,
    end: date,
    *,
    client: Minio,
    db_conn,
) -> None:
    """Build panel partitions for [start, end] and write to MinIO."""
    log.info("Building panel for %s → %s", start, end)

    # Load universe metadata
    log.info("Loading universe metadata from TimescaleDB ...")
    meta_df = _load_universe_metadata(db_conn)
    meta_map: dict[str, dict] = {}
    if not meta_df.empty:
        meta_map = meta_df.set_index("symbol").to_dict("index")

    # Discover symbols from MinIO bars/
    log.info("Listing symbols in bars/ ...")
    symbols = _list_bar_symbols(client)
    if not symbols:
        log.error("No symbols found in s3://%s/%s/", QID_EQUITY_BUCKET, BARS_PREFIX)
        sys.exit(1)
    log.info("Found %d symbols", len(symbols))

    # Load all raw bars into memory
    log.info("Loading raw OHLCV bars ...")
    symbol_bars = _load_all_bars(client, symbols, start, end)

    # Compute per-symbol features
    log.info("Computing per-symbol features ...")
    all_feature_dfs: list[pd.DataFrame] = []
    for i, (sym, raw_df) in enumerate(symbol_bars.items()):
        try:
            feat = _compute_symbol_features(raw_df)
        except Exception as exc:
            log.warning("Feature computation failed for %s: %s", sym, exc)
            continue

        feat["symbol"] = sym

        # Attach metadata
        m = meta_map.get(sym, {})
        feat["sector"] = m.get("sector")
        feat["industry"] = m.get("industry")
        feat["market_cap"] = m.get("market_cap")
        feat["is_etf"] = bool(m.get("is_etf", False))

        # Filter to [start, end] only (rolling windows required the warm-up)
        feat = feat[
            (feat["dt"] >= start) & (feat["dt"] <= end)
        ].reset_index(drop=True)
        if not feat.empty:
            all_feature_dfs.append(feat)

        if (i + 1) % 200 == 0:
            log.info("Feature computation: %d / %d symbols done", i + 1, len(symbol_bars))

    if not all_feature_dfs:
        log.error("No feature rows produced — aborting")
        sys.exit(1)

    log.info("Assembling cross-sectional panel ...")
    full_panel = pd.concat(all_feature_dfs, ignore_index=True)
    full_panel = full_panel.sort_values(["dt", "symbol"]).reset_index(drop=True)

    # Determine canonical column order
    canonical_cols = _canonical_columns()
    available = [c for c in canonical_cols if c in full_panel.columns]
    extra = [c for c in full_panel.columns if c not in canonical_cols]
    full_panel = full_panel[available + extra]

    # Write per-date partitions
    trading_dates = sorted(full_panel["dt"].unique())
    log.info("Writing %d daily partitions ...", len(trading_dates))
    written = 0
    for dt in trading_dates:
        slice_df = full_panel[full_panel["dt"] == dt].copy()
        if slice_df.empty:
            continue
        slice_df = _add_cross_sectional_features(slice_df)

        key = f"{PANEL_PREFIX}/dt={dt.isoformat()}/panel.parquet"
        try:
            _write_parquet_to_minio(client, key, slice_df)
            written += 1
        except Exception as exc:
            log.error("Failed to write partition %s: %s", dt, exc)

        if written % 100 == 0:
            log.info("Written %d / %d partitions", written, len(trading_dates))

    log.info("Done. Wrote %d partitions for %d symbols.", written, len(symbol_bars))


# ---------------------------------------------------------------------------
# Fill-rate validation (DuckDB sweep)
# ---------------------------------------------------------------------------

def validate_fill_rate(*, client: Minio) -> bool:
    """DuckDB sweep: check >=95% fill rate on all 4 new columns, 1y window.

    Returns True if all columns pass, False otherwise.
    """
    one_year_ago = (date.today() - timedelta(days=365)).isoformat()
    endpoint_proto = "https" if MINIO_SECURE else "http"
    endpoint_url = f"{endpoint_proto}://{MINIO_ENDPOINT}"

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        SET s3_endpoint='{MINIO_ENDPOINT}';
        SET s3_url_style='path';
        SET s3_use_ssl={'true' if MINIO_SECURE else 'false'};
        SET s3_access_key_id='{MINIO_ACCESS_KEY}';
        SET s3_secret_access_key='{MINIO_SECRET_KEY}';
    """)

    new_cols = [
        "bb_squeeze_ratio",
        "atr14_normalized_lag20",
        "range20_norm",
        "close_dev30_norm",
    ]
    all_pass = True
    for col in new_cols:
        result = con.execute(f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) AS filled,
                ROUND(100.0 * SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END)
                      / NULLIF(COUNT(*), 0), 2) AS fill_pct
            FROM read_parquet(
                's3://{QID_EQUITY_BUCKET}/{PANEL_PREFIX}/dt=*/panel.parquet',
                hive_partitioning=true
            )
            WHERE dt >= DATE '{one_year_ago}'
        """).fetchone()
        total, filled, fill_pct = result
        status = "PASS" if (fill_pct or 0) >= 95.0 else "FAIL"
        log.info("[%s] %s: %.2f%% fill (%d/%d rows)", status, col, fill_pct or 0.0, filled, total)
        if status == "FAIL":
            all_pass = False

    con.close()
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="qid-equity panel v1.1 builder")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument(
        "--cold-rebuild",
        action="store_true",
        help=f"Rebuild full 5y history from {COLD_REBUILD_START}",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run DuckDB fill-rate validation sweep and exit",
    )
    return p.parse_args()


def _canonical_columns() -> list[str]:
    """Canonical column order for the panel output."""
    return [
        # Identity
        "dt", "symbol", "sector", "industry", "market_cap", "is_etf",
        # OHLCV
        "open", "high", "low", "close", "volume", "trade_count", "vwap",
        # Returns
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        # Vol / momentum
        "realized_vol_20d", "realized_vol_60d", "realized_vol_60d_pctile",
        "atr14", "atr14_normalized", "rsi_14", "mom_z_5_60",
        # Trend distance
        "distance_from_20ma_pct", "distance_from_50ma_pct",
        "z_score_close_vs_20ma", "z_score_close_vs_50ma",
        # Breakout markers
        "pct_to_20d_high", "pct_to_60d_high", "high20_dist", "low20_dist",
        # Volume
        "dollar_volume", "dollar_volume_20d_avg",
        "volume_ratio_today", "volume_z_20",
        # Microstructure
        "gap_open", "range_pos", "vwap20_ratio",
        # Forward labels (ML)
        "fwd_return_1d", "fwd_return_5d", "fwd_return_10d", "fwd_return_20d",
        "fwd_max_runup_5d", "fwd_max_drawdown_5d",
        # Crucix overlay (NaN today)
        "n_ideas_24h", "n_ideas_7d",
        "theme_long_count_industry_7d", "theme_short_count_industry_7d",
        "industry_flow_rank", "sector_flow_rank",
        "earnings_date_proximity_days", "days_since_corporate_action",
        # New consolidation-gate columns (QIDP-231 / FIN-1145)
        "bb_squeeze_ratio",
        "atr14_normalized_lag20",
        "range20_norm",
        "close_dev30_norm",
    ]


def main() -> None:
    args = _parse_args()

    client = _minio_client()

    if args.validate:
        ok = validate_fill_rate(client=client)
        sys.exit(0 if ok else 1)

    if args.cold_rebuild:
        start = COLD_REBUILD_START
        end = date.today() - timedelta(days=1)
    elif args.start or args.end:
        start = date.fromisoformat(args.start) if args.start else COLD_REBUILD_START
        end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    else:
        # Daily incremental: yesterday
        yesterday = date.today() - timedelta(days=1)
        start = yesterday
        end = yesterday

    conn = psycopg2.connect(TSDB_DSN)
    try:
        build_panel(start, end, client=client, db_conn=conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

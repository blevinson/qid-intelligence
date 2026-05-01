#!/usr/bin/env python3
"""Build the daily panel parquet from per-symbol bar history.

QIDP-229 Sprint 1 — feature panel for TFARM-31's backtest. v1 schema:
bar-derived features only. Crucix overlay (tier B) and flow rank (tier C)
are added by separate jobs that UPSERT into the same partitioned layout.

Layout: qid-equity/panel/v1/dt=YYYY-MM-DD/panel.parquet
Schema: per (dt, symbol), strict no-leak — features at row t use only
data from rows ≤ t. Forward returns require future bars (set NaN if
unavailable).

Modes:
    cold   — build all dt partitions from full bar history (one-shot)
    daily  — build only the latest dt partition (cron)

Env:
    MINIO_*   endpoint, access/secret, bucket (default qid-equity)
    PANEL_MODE                 cold | daily (default daily)
    PANEL_LOOKBACK_DAYS        for daily mode, days back to (re)build (default 30 — handles late restatements)
    PANEL_SYMBOLS              optional CSV override
    PANEL_VERSION              parquet path subdir (default v1)
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from minio import Minio
from minio.error import S3Error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("panel_build")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio.qid.svc.cluster.local:9000")
MINIO_ACCESS = os.environ.get("MINIO_ACCESS_KEY", "").strip()
MINIO_SECRET = os.environ.get("MINIO_SECRET_KEY", "").strip()
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "qid-equity")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")

MODE = os.environ.get("PANEL_MODE", "daily").lower()
LOOKBACK = int(os.environ.get("PANEL_LOOKBACK_DAYS", "30"))
PANEL_VERSION = os.environ.get("PANEL_VERSION", "v1")
SYMBOLS_OVERRIDE = os.environ.get("PANEL_SYMBOLS", "").strip()

BARS_PREFIX = "bars/ohlcv-1d"
PANEL_PREFIX = f"panel/{PANEL_VERSION}"
UNIVERSE_KEY = "universe/snapshot.json"


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def _mc() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE)


def list_bar_symbols(client: Minio) -> list[str]:
    out = []
    for obj in client.list_objects(MINIO_BUCKET, prefix=BARS_PREFIX + "/", recursive=False):
        name = obj.object_name.rsplit("/", 1)[-1]
        if name.endswith(".parquet"):
            out.append(name[:-len(".parquet")])
    return sorted(out)


def load_universe_meta(client: Minio) -> pd.DataFrame:
    try:
        resp = client.get_object(MINIO_BUCKET, UNIVERSE_KEY)
        data = resp.read()
        resp.close()
        resp.release_conn()
    except S3Error as e:
        if e.code == "NoSuchKey":
            log.warning("no universe snapshot — rows will lack sector/industry/market_cap")
            return pd.DataFrame(columns=["symbol", "sector", "industry", "market_cap", "is_etf", "exchange"])
        raise
    payload = json.loads(data)
    rows = payload.get("symbols", [])
    df = pd.DataFrame(rows)
    keep = ["symbol", "sector", "industry", "market_cap", "is_etf", "exchange"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def read_bars(client: Minio, symbol: str) -> pd.DataFrame | None:
    try:
        resp = client.get_object(MINIO_BUCKET, f"{BARS_PREFIX}/{symbol}.parquet")
        data = resp.read()
        resp.close()
        resp.release_conn()
        df = pd.read_parquet(io.BytesIO(data))
        df["symbol"] = symbol
        df["dt"] = pd.to_datetime(df["dt"]).dt.date
        return df.sort_values("dt").reset_index(drop=True)
    except S3Error as e:
        if e.code == "NoSuchKey":
            return None
        raise


def write_partition(client: Minio, dt: date, df: pd.DataFrame) -> int:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    buf.seek(0)
    key = f"{PANEL_PREFIX}/dt={dt.isoformat()}/panel.parquet"
    client.put_object(
        MINIO_BUCKET, key, buf, length=len(buf.getvalue()),
        content_type="application/octet-stream",
    )
    return len(buf.getvalue())


# ---------------------------------------------------------------------------
# Feature computation (no-leak — every feature at row t uses only rows ≤ t)
# ---------------------------------------------------------------------------

def featurize_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bar-derived features for one symbol's full history."""
    if df.empty:
        return df
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    open_ = out["open"]
    vol = out["volume"]

    # Returns
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)

    # Volatility / range
    out["realized_vol_20d"] = out["ret_1d"].rolling(20, min_periods=10).std() * np.sqrt(252)
    out["realized_vol_60d"] = out["ret_1d"].rolling(60, min_periods=30).std() * np.sqrt(252)
    # Percentile of realized_vol_60d in the trailing 252d window — flat-detector core
    out["realized_vol_60d_pctile"] = (
        out["realized_vol_60d"]
        .rolling(252, min_periods=60)
        .apply(lambda s: (s.iloc[-1] <= s).mean() if not np.isnan(s.iloc[-1]) else np.nan, raw=False)
    )

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr14"] = atr14
    out["atr14_normalized"] = atr14 / close

    # RSI 14
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Momentum z (5d return z-scored on 60d window)
    ret5 = close.pct_change(5)
    out["mom_z_5_60"] = (ret5 - ret5.rolling(60, min_periods=30).mean()) / ret5.rolling(60, min_periods=30).std()

    # Distance from MAs
    ma20 = close.rolling(20, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=20).mean()
    out["distance_from_20ma_pct"] = (close - ma20) / ma20
    out["distance_from_50ma_pct"] = (close - ma50) / ma50

    # Mean-reversion z-scores
    std20 = close.rolling(20, min_periods=10).std()
    std50 = close.rolling(50, min_periods=20).std()
    out["z_score_close_vs_20ma"] = (close - ma20) / std20
    out["z_score_close_vs_50ma"] = (close - ma50) / std50

    # Breakout markers (no-leak: max excludes today)
    high20_excl = close.shift(1).rolling(20, min_periods=10).max()
    high60_excl = close.shift(1).rolling(60, min_periods=30).max()
    low20_excl = close.shift(1).rolling(20, min_periods=10).min()
    out["pct_to_20d_high"] = (high20_excl - close) / close       # ≤0 means already broken out
    out["pct_to_60d_high"] = (high60_excl - close) / close
    out["high20_dist"] = (close - high20_excl) / high20_excl     # +N% above prior 20d high
    out["low20_dist"] = (close - low20_excl) / low20_excl

    # Volume / flow (bar-level)
    out["dollar_volume"] = close * vol
    out["dollar_volume_20d_avg"] = out["dollar_volume"].rolling(20, min_periods=10).mean()
    out["volume_ratio_today"] = vol / vol.rolling(20, min_periods=10).mean()
    out["volume_z_20"] = (vol - vol.rolling(20, min_periods=10).mean()) / vol.rolling(20, min_periods=10).std()

    # Open / range positioning
    out["gap_open"] = (open_ - prev_close) / prev_close
    out["range_pos"] = (close - low) / (high - low).replace(0, np.nan)

    vwap20 = (out.get("vwap", close) * vol).rolling(20, min_periods=10).sum() / vol.rolling(20, min_periods=10).sum()
    out["vwap20_ratio"] = close / vwap20

    # Forward returns (LABELS — strict shift, NaN at panel-tail)
    out["fwd_return_1d"] = close.shift(-1) / close - 1
    out["fwd_return_5d"] = close.shift(-5) / close - 1
    out["fwd_return_10d"] = close.shift(-10) / close - 1
    out["fwd_return_20d"] = close.shift(-20) / close - 1

    # Forward intra-window extremes (for stop / take-profit calibration)
    rolling_max_5 = close.shift(-5).rolling(5, min_periods=1).max()  # placeholder — see note
    rolling_min_5 = close.shift(-5).rolling(5, min_periods=1).min()
    # The above isn't quite right; compute via rolling on reversed series:
    fwd_window = 5
    out["fwd_max_runup_5d"] = (
        close.iloc[::-1].rolling(fwd_window, min_periods=1).max().iloc[::-1].shift(-1) / close - 1
    )
    out["fwd_max_drawdown_5d"] = (
        close.iloc[::-1].rolling(fwd_window, min_periods=1).min().iloc[::-1].shift(-1) / close - 1
    )

    return out


# Crucix-derived columns (tier B) and flow-rank columns (tier C) reserved as
# NULL placeholders so downstream consumers see a stable schema. Separate
# jobs UPSERT these later.
CRUCIX_PLACEHOLDERS = [
    "n_ideas_24h", "n_ideas_7d",
    "theme_long_count_industry_7d", "theme_short_count_industry_7d",
    "industry_flow_rank", "sector_flow_rank",
    "earnings_date_proximity_days", "days_since_corporate_action",
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_panel(symbols: list[str], dts_to_write: set[date] | None) -> dict:
    client = _mc()
    if not client.bucket_exists(MINIO_BUCKET):
        raise RuntimeError(f"bucket {MINIO_BUCKET} missing")

    universe = load_universe_meta(client)
    universe_lookup = universe.set_index("symbol").to_dict("index") if not universe.empty else {}

    log.info("featurizing %d symbols", len(symbols))
    per_symbol_frames: list[pd.DataFrame] = []
    for i, sym in enumerate(symbols, start=1):
        bars = read_bars(client, sym)
        if bars is None or bars.empty:
            continue
        feat = featurize_symbol(bars)
        # Carry universe meta forward
        meta = universe_lookup.get(sym, {})
        feat["sector"] = meta.get("sector")
        feat["industry"] = meta.get("industry")
        feat["market_cap"] = meta.get("market_cap")
        feat["is_etf"] = bool(meta.get("is_etf")) if meta.get("is_etf") is not None else None
        # Tier B/C placeholders (NaN by design)
        for col in CRUCIX_PLACEHOLDERS:
            feat[col] = np.nan
        per_symbol_frames.append(feat)
        if i % 200 == 0:
            log.info("  %d / %d symbols featurized", i, len(symbols))

    if not per_symbol_frames:
        log.warning("no bar data found — exiting")
        return {"partitions": 0, "rows": 0}

    panel = pd.concat(per_symbol_frames, ignore_index=True)
    log.info("panel: %d total rows pre-filter", len(panel))

    # Filter to requested dts (or all)
    if dts_to_write is not None:
        panel = panel[panel["dt"].isin(dts_to_write)]
        log.info("filtered to %d rows across %d target dts", len(panel), len(dts_to_write))

    # Stable column order
    column_order = (
        ["dt", "symbol", "sector", "industry", "market_cap", "is_etf",
         "open", "high", "low", "close", "volume", "trade_count", "vwap",
         "ret_1d", "ret_5d", "ret_10d", "ret_20d",
         "realized_vol_20d", "realized_vol_60d", "realized_vol_60d_pctile",
         "atr14", "atr14_normalized",
         "rsi_14", "mom_z_5_60",
         "distance_from_20ma_pct", "distance_from_50ma_pct",
         "z_score_close_vs_20ma", "z_score_close_vs_50ma",
         "pct_to_20d_high", "pct_to_60d_high", "high20_dist", "low20_dist",
         "dollar_volume", "dollar_volume_20d_avg",
         "volume_ratio_today", "volume_z_20",
         "gap_open", "range_pos", "vwap20_ratio",
         "fwd_return_1d", "fwd_return_5d", "fwd_return_10d", "fwd_return_20d",
         "fwd_max_runup_5d", "fwd_max_drawdown_5d"]
        + CRUCIX_PLACEHOLDERS
    )
    column_order = [c for c in column_order if c in panel.columns]
    panel = panel[column_order]

    n_partitions = 0
    n_rows = 0
    by_dt = panel.groupby("dt", sort=True)
    for dt, slice_ in by_dt:
        slice_ = slice_.drop(columns=["dt"]).copy()
        # Re-add dt as the partition column on the outside; parquet stays
        # without it (path encodes it). Keep dt INSIDE for self-contained reads.
        slice_["dt"] = dt
        slice_ = slice_[["dt", "symbol"] + [c for c in column_order if c not in ("dt", "symbol")]]
        nbytes = write_partition(client, dt, slice_)
        n_partitions += 1
        n_rows += len(slice_)
        if n_partitions % 50 == 0:
            log.info("  wrote %d partitions (%d rows total)", n_partitions, n_rows)

    log.info("done: %d partitions, %d total rows", n_partitions, n_rows)
    return {"partitions": n_partitions, "rows": n_rows}


def main() -> int:
    if not MINIO_ACCESS or not MINIO_SECRET:
        log.error("MINIO_ACCESS_KEY / MINIO_SECRET_KEY required")
        return 2

    client = _mc()
    if SYMBOLS_OVERRIDE:
        symbols = sorted({s.strip().upper() for s in SYMBOLS_OVERRIDE.split(",") if s.strip()})
    else:
        symbols = list_bar_symbols(client)
    log.info("MODE=%s  symbols=%d  lookback=%d", MODE, len(symbols), LOOKBACK)

    if MODE == "cold":
        # All dts present in any bar parquet → write each partition
        result = build_panel(symbols, dts_to_write=None)
    elif MODE == "daily":
        cutoff = date.today() - timedelta(days=LOOKBACK)
        # Restrict to bars from cutoff forward — featurize still needs history
        # so we read full bars but only WRITE partitions for dt >= cutoff
        target_dts: set[date] = set()
        for sym in symbols:
            df = read_bars(client, sym)
            if df is None or df.empty:
                continue
            target_dts.update([d for d in df["dt"].unique() if d >= cutoff])
        log.info("daily mode → %d target dts (>= %s)", len(target_dts), cutoff)
        result = build_panel(symbols, dts_to_write=target_dts)
    else:
        log.error("PANEL_MODE must be cold or daily; got %r", MODE)
        return 2

    log.info("result: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

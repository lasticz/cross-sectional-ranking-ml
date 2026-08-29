# -*- coding: utf-8 -*-
"""
ML data pipeline: 27-coin 15m futures K-lines + funding rates -> cleaned,
cross-sectionally standardized long-format dataset.

Steps
-----
1. Load 15m klines + 1h funding rates for every pair in config.json
   (exchange.pair_whitelist), clipped to [start, end).
2. Clean per coin:
   a. 3-sigma outlier removal, OHLCV columns independently, applied to
      LOG-RETURNS with rolling-window statistics (default 7d of 15m bars).
      Sigma is computed on a rolling window rather than globally so that
      multi-year price trends are not mistaken for outliers, and on returns
      rather than levels so that ordinary drift is not flagged (a level-based
      test flags ~3% of bars and would corrupt the data). Flagged -> NaN.
   b. Reindex to the coin's own full 15m grid (missing bars -> NaN rows),
      then forward-fill with limit (default 4 bars). Longer gaps stay NaN.
   c. Drop zero-volume bars (non-trading sessions).
3. Merge funding rate (last known value, backward as-of) onto each 15m bar.
4. Cross-sectional z-score: for each timestamp, each feature column is
   standardized across all coins present at that timestamp. This removes the
   "BTC 60000 vs DOGE 0.1" scale problem; the model learns each coin's state
   relative to the universe at that moment.
5. Save:
   - user_data/ml/clean_data.feather   (long DataFrame)
   - user_data/ml/data_stats.json      (per-column stats / QC metrics)

Output columns: date, base, open, high, low, close, volume, funding,
                _cs_open, _cs_high, _cs_low, _cs_close, _cs_volume, _cs_funding

Usage
-----
    python scripts/ml/data_pipeline.py [--window 672] [--sigma 3.0] ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()  # 项目根（从项目根目录运行: python scripts/ml/xxx.py）
DATA_DIR = ROOT / "user_data" / "data" / "binance" / "futures"
OUT_DIR = ROOT / "user_data" / "ml"
DEFAULT_CONFIG = ROOT / "config.json"

OHLCV = ["open", "high", "low", "close", "volume"]
FEATURES = ["open", "high", "low", "close", "volume", "funding"]

DATE_DTYPE = "datetime64[ns, UTC]"  # uniform resolution: merge_asof requires matching keys


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_pairs(config_path: Path) -> list[str]:
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    pairs = cfg["exchange"]["pair_whitelist"]
    bases = [p.split("/")[0] for p in pairs]
    if len(set(bases)) != len(bases):
        raise ValueError("duplicate pairs in config whitelist")
    return bases


def load_klines(base: str, start: str, end: str) -> pd.DataFrame:
    path = DATA_DIR / f"{base}_USDT_USDT-15m-futures.feather"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_feather(path)
    df = df[["date", *OHLCV]].copy()
    df["date"] = df["date"].astype(DATE_DTYPE)
    df = df[(df["date"] >= start) & (df["date"] < end)]
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
    return df.reset_index(drop=True)


def load_funding(base: str, start: str, end: str) -> pd.DataFrame:
    """Funding-rate feather has OHLCV layout; the rate lives in `open`."""
    path = DATA_DIR / f"{base}_USDT_USDT-1h-funding_rate.feather"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_feather(path)
    df = df[["date", "open"]].rename(columns={"open": "funding"})
    df["date"] = df["date"].astype(DATE_DTYPE)
    # keep one funding rate ahead of the window so the first bars get a value
    df = df[df["date"] < end]
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def remove_outliers_3sigma(
    df: pd.DataFrame, window: int, min_periods: int, sigma: float, base: str,
    counters: dict,
) -> pd.DataFrame:
    """3-sigma outlier screen, OHLCV columns independently, on LOG-RETURNS.

    A 3-sigma test on raw price LEVELS misfires: normal multi-day drift pushes
    ~3% of bars beyond 3 sigma of a rolling-window mean, which would shred the
    dataset. Bad ticks instead show up as return spikes, so we flag
    |log(x_t / x_{t-1}) - rolling_mean| > sigma * rolling_std and null the
    level; the ffill step afterwards repairs isolated flagged bars.
    """
    df = df.copy()
    for col in OHLCV:
        x = df[col]
        r = np.log(x.where(x > 0)).diff()
        roll = r.rolling(window=window, min_periods=min_periods)
        mu = roll.mean()
        sd = roll.std()
        mask = ((r - mu).abs() > sigma * sd) & r.notna()
        mask = mask.fillna(False)
        n_flag = int(mask.sum())
        counters["outliers"][col] += n_flag
        if n_flag:
            df.loc[mask, col] = np.nan
    return df


def reindex_ffill(df: pd.DataFrame, limit: int, base: str, counters: dict) -> pd.DataFrame:
    """Expand to the coin's full 15m grid, then forward-fill up to `limit` bars."""
    if df.empty:
        return df
    full = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="15min")
    full = full.tz_localize("UTC") if full.tz is None else full
    df = df.set_index("date").reindex(full)
    df.index.name = "date"
    counters["gap_rows"] += int(df[OHLCV].isna().all(axis=1).sum())
    df = df.ffill(limit=limit)
    return df.reset_index()


def drop_zero_volume(df: pd.DataFrame, base: str, counters: dict) -> pd.DataFrame:
    mask = df["volume"] == 0  # NaN volume rows are NOT dropped (marked NaN)
    counters["zero_volume_dropped"] += int(mask.fillna(False).sum())
    return df[~mask.fillna(False)].reset_index(drop=True)


def merge_funding(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    if funding.empty:
        klines["funding"] = np.nan
        return klines
    out = pd.merge_asof(
        klines.sort_values("date"), funding.sort_values("date"),
        on="date", direction="backward",
    )
    return out


def process_coin(
    base: str, start: str, end: str,
    window: int, min_periods: int, sigma: float, ffill_limit: int,
    counters: dict,
) -> pd.DataFrame:
    kl = load_klines(base, start, end)
    if kl.empty:
        print(f"  [WARN] {base}: no kline rows in range, skipped")
        return pd.DataFrame()
    counters["raw_rows"] += len(kl)

    kl = remove_outliers_3sigma(kl, window, min_periods, sigma, base, counters)
    kl = reindex_ffill(kl, ffill_limit, base, counters)
    kl = drop_zero_volume(kl, base, counters)

    fund = load_funding(base, start, end)
    kl = merge_funding(kl, fund)
    kl["base"] = base
    return kl[["date", "base", *FEATURES]]


# --------------------------------------------------------------------------- #
# Cross-sectional standardization
# --------------------------------------------------------------------------- #
def cross_sectional_zscore(long_df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """z-score each feature across coins per timestamp. Wide-pivot based."""
    long_df = long_df.sort_values(["date", "base"]).reset_index(drop=True)
    mi = pd.MultiIndex.from_frame(long_df[["date", "base"]])
    for col in FEATURES:
        wide = long_df.pivot(index="date", columns="base", values=col)
        wv = wide.to_numpy(dtype=np.float64)
        cnt = np.sum(~np.isnan(wv), axis=1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            mu = np.nanmean(wv, axis=1)
            sd = np.nanstd(wv, axis=1, ddof=1)
        mu = np.where(np.isfinite(mu), mu, np.nan)
        sd = np.where(np.isfinite(sd), sd, np.nan)

        z = (wv - mu[:, None]) / sd[:, None]          # 0/0 and x/0 -> NaN/inf
        z = np.where(np.isfinite(z), z, np.nan)
        # degenerate cross-sections (all coins identical): neutral 0.0
        flat = (np.nan_to_num(sd, nan=np.inf) <= 1e-12) & (cnt > 0)
        z[flat] = np.where(np.isnan(wv[flat]), np.nan, 0.0)
        # too few coins at this timestamp -> no reliable cross-section
        z[cnt < min_count, :] = np.nan

        cart = pd.MultiIndex.from_product([wide.index, wide.columns])
        zs = pd.Series(z.ravel(order="C"), index=cart).reindex(mi)
        long_df[f"_cs_{col}"] = zs.to_numpy()
    return long_df


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def build_stats(
    long_df: pd.DataFrame, counters: dict, args: argparse.Namespace,
    bases: list[str], raw_rows: int,
) -> dict:
    col_stats: dict[str, dict] = {}
    for col in FEATURES:
        s = long_df[col]
        n_out = counters["outliers"].get(col)  # funding is not outlier-screened
        col_stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "missing_rate": float(s.isna().mean()),
            "outlier_rate": float(n_out / raw_rows) if (n_out and raw_rows) else 0.0,
        }
    for col in FEATURES:
        s = long_df[f"_cs_{col}"]
        col_stats[f"_cs_{col}"] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "missing_rate": float(s.isna().mean()),
            "outlier_rate": None,  # not applicable to standardized output
        }
    per_pair = {}
    for base, g in long_df.groupby("base"):
        per_pair[base] = {
            "rows": int(len(g)),
            "start": str(g["date"].iloc[0]),
            "end": str(g["date"].iloc[-1]),
            "nan_rows_ohlcv": int(g[OHLCV].isna().all(axis=1).sum()),
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "start": args.start, "end_exclusive": args.end,
            "n_pairs": len(bases), "pairs": bases,
            "rolling_window_bars": args.window,
            "rolling_min_periods": args.min_periods,
            "sigma": args.sigma, "ffill_limit": args.ffill_limit,
            "min_cs_count": args.min_cs_count,
        },
        "overall": {
            "rows": int(len(long_df)),
            "raw_rows_loaded": int(raw_rows),
            "date_min": str(long_df["date"].min()),
            "date_max": str(long_df["date"].max()),
            "zero_volume_dropped": int(counters["zero_volume_dropped"]),
            "outliers_flagged_total": int(sum(counters["outliers"].values())),
        },
        "columns": col_stats,
        "per_pair": per_pair,
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(long_df: pd.DataFrame) -> None:
    print("\n===== VALIDATION =====")
    print(f"rows: {len(long_df):,}   coins: {long_df['base'].nunique()}")
    print(f"date range: {long_df['date'].min()} -> {long_df['date'].max()}")
    print(f"columns: {list(long_df.columns)}")
    nan_pct = long_df.isna().mean() * 100
    print("NaN% per column:")
    for c, v in nan_pct.items():
        print(f"  {c:<14s} {v:8.4f}%")
    all_nan = [c for c in long_df.columns if long_df[c].isna().all()]
    if all_nan:
        raise RuntimeError(f"all-NaN columns found: {all_nan}")
    # cross-sectional sanity: per timestamp, _cs_* mean ~ 0
    for col in ["_cs_open", "_cs_close", "_cs_volume", "_cs_funding"]:
        ts_mean = long_df.groupby("date")[col].mean()
        ts_std = long_df.groupby("date")[col].std()
        print(
            f"  {col}: per-timestamp mean max|.|={ts_mean.abs().max():.2e}, "
            f"per-timestamp std median={ts_std.median():.4f}"
        )
    dup = long_df.duplicated(subset=["date", "base"]).sum()
    if dup:
        raise RuntimeError(f"{dup} duplicate (date, base) rows")
    print("OK: no all-NaN columns, no duplicate (date, base).")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-08-28", help="exclusive")
    p.add_argument("--window", type=int, default=672, help="rolling 3-sigma window in 15m bars (672 = 7d), applied to log-returns")
    p.add_argument("--min-periods", type=int, default=96, help="min bars before outlier flagging (96 = 1d)")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--ffill-limit", type=int, default=4)
    p.add_argument("--min-cs-count", type=int, default=5,
                   help="min coins present at a timestamp to compute cross-sectional z")
    return p.parse_args(argv)


def main(argv=None) -> int:
    t0 = time.time()
    args = parse_args(argv)

    bases = load_pairs(args.config)
    print(f"pairs from config: {len(bases)} -> {bases}")

    counters = {
        "raw_rows": 0,
        "zero_volume_dropped": 0,
        "gap_rows": 0,
        "outliers": {c: 0 for c in OHLCV},
    }

    frames = []
    for base in bases:
        df = process_coin(base, args.start, args.end,
                          args.window, args.min_periods, args.sigma,
                          args.ffill_limit, counters)
        if not df.empty:
            frames.append(df)
        print(f"  {base:<10s} rows={len(df):>7,}")

    long_df = pd.concat(frames, ignore_index=True)
    print(f"\ncombined rows before cs-zscore: {len(long_df):,}")

    long_df = cross_sectional_zscore(long_df, args.min_cs_count)

    raw_rows = counters["raw_rows"]
    stats = build_stats(long_df, counters, args, bases, raw_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_feather = OUT_DIR / "clean_data.feather"
    out_json = OUT_DIR / "data_stats.json"
    long_df.to_feather(out_feather)
    out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nsaved: {out_feather}  ({out_feather.stat().st_size / 1e6:.1f} MB)")
    print(f"saved: {out_json}")
    validate(long_df)
    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

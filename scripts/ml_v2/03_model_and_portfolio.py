# -*- coding: utf-8 -*-
"""ML v2 Phase 3+4: 模型训练 + Top3/Bottom3 组合回测

Phase 3: LightGBM 回归 label_neutral, walk-forward (12月训练→1月预测)
Phase 4: 每 4h rebalance, 做多预测 Top3 + 做空 Bottom3, 含成本敏感性

用法: python scripts/ml_v2/03_model_and_portfolio.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

# 路径: 从本文件出发定位项目根目录（不使用 ../）
ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/ml_v2/ → 项目根
ML_DIR = ROOT / "user_data" / "ml_v2"
FEE_RT = 0.0004  # maker 4bps 往返


def load_data():
    df = pd.read_feather(ML_DIR / "feature_matrix.feather")
    meta_cols = {"date", "base", "label_abs", "label_neutral", "label_rank"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    print(f"数据: {len(df)} 行 × {len(feat_cols)} 特征 | {df['date'].min()} → {df['date'].max()}")
    return df, feat_cols


def walk_forward_predict(df, feat_cols, target_col="label_neutral"):
    """12个月训练 → 1月预测, 输出全期间 OOS 预测"""
    months = df["date"].dt.to_period("M")
    predictions = []
    all_months = sorted(months.unique())

    for i in range(12, len(all_months)):
        test_month = all_months[i]
        train_start = all_months[i - 12]
        train_end = all_months[i - 1]
        tr_mask = (months >= train_start) & (months <= train_end)
        te_mask = months == test_month
        if tr_mask.sum() < 10000 or te_mask.sum() < 100:
            continue

        med = df.loc[tr_mask, feat_cols].median(numeric_only=True)
        X_tr = df.loc[tr_mask, feat_cols].fillna(med).to_numpy(dtype=np.float32)
        y_tr = df.loc[tr_mask, target_col].to_numpy(dtype=np.float32)
        valid = np.isfinite(y_tr)
        X_tr, y_tr = X_tr[valid], y_tr[valid]

        X_te = df.loc[te_mask, feat_cols].fillna(med).to_numpy(dtype=np.float32)
        te_idx = df.index[te_mask]

        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=7, learning_rate=0.03,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1, force_col_wise=True, n_jobs=4,
        )
        model.fit(X_tr, y_tr)
        predictions.append(pd.DataFrame({"idx": te_idx, "pred": model.predict(X_te)}))

        if i % 6 == 0:
            print(f"  fold {i-11}: {test_month} (train {train_start}~{train_end})", flush=True)

    all_pred = pd.concat(predictions).set_index("idx")
    df["pred"] = np.nan
    df.loc[all_pred.index, "pred"] = all_pred["pred"]
    return df


def compute_rank_ic(df):
    valid = df.dropna(subset=["pred", "label_neutral"])
    return valid.groupby("date").apply(
        lambda g: g["pred"].corr(g["label_neutral"], method="spearman")
        if len(g) >= 10 else np.nan,
        include_groups=False,
    ).dropna()


def portfolio_backtest(df, rebalance_bars=16, top_n=3):
    valid = df.dropna(subset=["pred"]).copy()
    dates = sorted(valid["date"].unique())
    decision_dates = dates[::rebalance_bars]
    trades = []
    prev_long, prev_short = set(), set()

    for dd in decision_dates:
        snap = valid[valid["date"] == dd]
        if len(snap) < 10:
            continue
        snap = snap.sort_values("pred", ascending=False)
        long_set = set(snap.head(top_n)["base"])
        short_set = set(snap.tail(top_n)["base"])
        future = snap.set_index("base")["label_neutral"]

        l_rets = [future[b] for b in long_set if b in future.index and np.isfinite(future[b])]
        s_rets = [future[b] for b in short_set if b in future.index and np.isfinite(future[b])]
        if not l_rets or not s_rets:
            continue

        long_ret = np.mean(l_rets)
        short_ret = np.mean(s_rets)
        to_long = len(long_set - prev_long) / top_n
        to_short = len(short_set - prev_short) / top_n
        turnover = (to_long + to_short) / 2

        trades.append({
            "date": dd, "ls_ret": long_ret - short_ret, "lo_ret": long_ret,
            "turnover": turnover, "to_long": to_long, "to_short": to_short,
        })
        prev_long, prev_short = long_set, short_set

    return pd.DataFrame(trades)


def main():
    t0 = time.time()
    print("=== Phase 3: Walk-Forward 模型训练 ===\n")
    df, feat_cols = load_data()
    df = walk_forward_predict(df, feat_cols)

    print("\n=== Rank IC 分析 ===")
    ic = compute_rank_ic(df)
    ic_ir = ic.mean() / ic.std() if ic.std() > 0 else 0
    print(f"时间戳: {len(ic)} | mean IC: {ic.mean():.4f} | std: {ic.std():.4f}")
    print(f"IC IR: {ic_ir:.4f} | t-stat: {ic_ir * np.sqrt(len(ic)):.1f}")
    print(f"IC > 0 比例: {(ic > 0).mean() * 100:.1f}%")

    print(f"\n=== Phase 4: Top3/Bottom3 组合回测 ===\n")
    portfolio = portfolio_backtest(df)
    n_year = 365 * 24 / 4
    print(f"rebalance 次数: {len(portfolio)} | 平均换手率: {portfolio['turnover'].mean() * 100:.1f}%")

    print("\n成本敏感性:")
    print(f"{'成本':12s} {'L+S毛利':>10s} {'L+S净':>10s} {'L-only毛利':>10s} {'L-only净':>10s}")
    for label, mult in [("1x(4bps)", 1.0), ("1.5x(6bps)", 1.5), ("2x(8bps)", 2.0)]:
        cost = portfolio["turnover"] * FEE_RT * mult
        ls_g = portfolio["ls_ret"].mean() * n_year * 100
        lo_g = portfolio["lo_ret"].mean() * n_year * 100
        ls_n = (portfolio["ls_ret"] - cost).mean() * n_year * 100
        lo_n = (portfolio["lo_ret"] - cost * 0.5).mean() * n_year * 100
        print(f"{label:12s} {ls_g:+9.1f}% {ls_n:+9.1f}% {lo_g:+9.1f}% {lo_n:+9.1f}%")

    # 十分位
    print("\n=== 十分位收益 ===")
    valid = df.dropna(subset=["pred", "label_neutral"]).copy()
    valid["decile"] = valid.groupby("date")["pred"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    dec = valid.groupby("decile")["label_neutral"].mean() * 10000
    for d in sorted(valid["decile"].dropna().unique()):
        if d in dec.index:
            print(f"  D{int(d)+1:2d}: {dec[d]:+7.1f}bps")
    spread = dec.iloc[-1] - dec.iloc[0] if len(dec) >= 2 else 0
    print(f"  D10-D1 spread: {spread:+.1f}bps/4h")

    portfolio.to_feather(ML_DIR / "portfolio_trades.feather")
    summary = {
        "rank_ic_mean": float(ic.mean()), "rank_ic_ir": float(ic_ir),
        "d10_d1_spread_bps": float(spread),
        "avg_turnover": float(portfolio["turnover"].mean()),
    }
    (ML_DIR / "portfolio_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""ML v2 → freqtrade 信号桥: walk-forward 预测 → Top3/Bottom3 信号文件

与 copy-trade 的信号桥同一模式: 生成 freqtrade 策略可读取的信号文件,
由 freqtrade 的执行引擎处理入场/离场/资金费/强平/仓位管理。

用法: python scripts/ml_v2/04_freqtrade_bridge.py
输出: user_data/ml_signals.json (freqtrade 策略读取)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path.cwd()
ML_DIR = ROOT / "user_data" / "ml_v2"
SIGNAL_FILE = ROOT / "user_data" / "ml_signals.json"
REBALANCE_BARS = 16  # 4h
TOP_N = 3


def load_and_predict():
    """加载特征矩阵, 走 walk-forward, 返回带预测列的 DataFrame"""
    df = pd.read_feather(ML_DIR / "feature_matrix.feather")
    meta_cols = {"date", "base", "label_abs", "label_neutral", "label_rank"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df["date"] = pd.to_datetime(df["date"], utc=True)

    months = df["date"].dt.to_period("M")
    predictions = []
    all_months = sorted(months.unique())

    for i in range(12, len(all_months)):
        test_month = all_months[i]
        train_start = all_months[i - 12]
        tr_mask = (months >= train_start) & (months <= all_months[i - 1])
        te_mask = months == test_month
        if tr_mask.sum() < 10000 or te_mask.sum() < 100:
            continue

        med = df.loc[tr_mask, feat_cols].median(numeric_only=True)
        X_tr = df.loc[tr_mask, feat_cols].fillna(med).to_numpy(dtype=np.float32)
        y_tr = df.loc[tr_mask, "label_neutral"].to_numpy(dtype=np.float32)
        valid = np.isfinite(y_tr)
        X_tr, y_tr = X_tr[valid], y_tr[valid]
        X_te = df.loc[te_mask, feat_cols].fillna(med).to_numpy(dtype=np.float32)

        model = lgb.LGBMRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, force_col_wise=True, n_jobs=4,
        )
        model.fit(X_tr, y_tr)
        predictions.append(pd.DataFrame({
            "idx": df.index[te_mask], "pred": model.predict(X_te),
            "date": df.loc[te_mask, "date"].to_numpy(),
            "base": df.loc[te_mask, "base"].to_numpy(),
        }))
        if i % 6 == 0:
            print(f"  fold {i-11}: {test_month}", flush=True)

    return pd.concat(predictions, ignore_index=True)


def generate_signals(pred_df):
    """从预测中提取每 4h 的 Top3/Bottom3, 生成 freqtrade 信号"""
    # base 列格式转换: 1000PEPE_USDT_USDT → 1000PEPE/USDT:USDT
    def to_pair(base):
        if "/" in base:
            return base  # 已是 pair 格式
        parts = base.split("_USDT_USDT")
        if parts[0] != base:
            return parts[0] + "/USDT:USDT"
        return base + "/USDT:USDT"  # fallback

    # 按 date 分组, 每 REBALANCE_BARS 个 bar 取一个决策点
    unique_dates = sorted(pred_df["date"].unique())
    decision_dates = unique_dates[::REBALANCE_BARS]
    decision_set = set(decision_dates)

    signals = []
    for dd in decision_dates:
        snap = pred_df[pred_df["date"] == dd].sort_values("pred", ascending=False)
        if len(snap) < 10:
            continue

        top = snap.head(TOP_N)
        bottom = snap.tail(TOP_N)
        # 信号窗口: 从当前决策点到下一个决策点
        try:
            next_dd = decision_dates[decision_dates.index(dd) + 1]
        except (ValueError, IndexError):
            next_dd = pd.Timestamp(dd) + pd.Timedelta(hours=4)

        for _, row in top.iterrows():
            signals.append({
                "pair": to_pair(row["base"]),
                "window_start": str(pd.Timestamp(dd).isoformat()),
                "window_end": str(pd.Timestamp(next_dd).isoformat()),
                "side": "long",
                "hold_h": 4,
                "wtype": "ml_top3",
                "pred": float(row["pred"]),
            })
        for _, row in bottom.iterrows():
            signals.append({
                "pair": to_pair(row["base"]),
                "window_start": str(pd.Timestamp(dd).isoformat()),
                "window_end": str(pd.Timestamp(next_dd).isoformat()),
                "side": "short",
                "hold_h": 4,
                "wtype": "ml_bottom3",
                "pred": float(row["pred"]),
            })

    return signals


def main():
    t0 = time.time()
    print("=== Walk-Forward 预测 ===")
    pred_df = load_and_predict()
    print(f"预测完成: {len(pred_df)} 行, 耗时 {time.time()-t0:.0f}s")

    print("\n=== 生成信号文件 ===")
    signals = generate_signals(pred_df)
    SIGNAL_FILE.write_text(json.dumps(signals, indent=1), encoding="utf-8")
    longs = sum(1 for s in signals if s["side"] == "long")
    shorts = sum(1 for s in signals if s["side"] == "short")
    print(f"信号: {len(signals)} 条 (long={longs}, short={shorts})")
    print(f"保存到: {SIGNAL_FILE}")

    # 统计
    dates = sorted(set(s["window_start"] for s in signals))
    print(f"决策点: {len(dates)} 个, 从 {dates[0]} 到 {dates[-1]}")


if __name__ == "__main__":
    main()

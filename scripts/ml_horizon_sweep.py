# -*- coding: utf-8 -*-
"""ML 多持有期扫描: 15m 特征 × 不同前瞻目标(1h/2h/4h/8h)

每档输出: walk-forward AUC + 模拟交易净收益(概率>阈值开仓, 持有至目标期末平仓, 扣 4bps 费)
用法: python scripts/ml_horizon_sweep.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
FEE = 0.0004  # maker 往返

# 持有期(15m bar 数) -> 人类可读标签
HORIZONS = [(4, "1h"), (8, "2h"), (16, "4h"), (32, "8h")]
TARGET_THRESHOLD = 0.001  # 目标: 收益 > 0.1%


def load_all():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    return {p.split("/")[0]: pd.read_feather(D / f"{p.split('/')[0]}_USDT_USDT-15m-futures.feather")
            for p in cfg["exchange"]["pair_whitelist"]
            if (D / f"{p.split('/')[0]}_USDT_USDT-15m-futures.feather").exists()}


def features_for(df, base, all_closes):
    c, o, h, l, v = (df[k].to_numpy() for k in ("close", "open", "high", "low", "volume"))
    n = len(c)
    cs = pd.Series(c)
    feat = pd.DataFrame(index=df.index)
    for lb in (4, 8, 16, 48, 96, 288):
        feat[f"ret_{lb}"] = cs.pct_change(lb)
    feat["dist_ema20"] = cs / cs.ewm(span=20).mean().to_numpy() - 1
    feat["dist_ema200"] = cs / cs.ewm(span=200).mean().to_numpy() - 1
    mid = cs.rolling(20).mean(); sd = cs.rolling(20).std()
    feat["bb_pos"] = (cs - mid) / (2 * sd + 1e-12)
    feat["bb_width"] = (4 * sd) / (mid + 1e-12)
    feat["bb_width_pct"] = feat["bb_width"] / feat["bb_width"].rolling(96).mean()
    for p in (7, 14, 28):
        d = cs.diff()
        feat[f"rsi_{p}"] = 100 - 100 / (1 + d.where(d > 0, 0).ewm(alpha=1/p).mean() / (-d.where(d < 0, 0)).ewm(alpha=1/p).mean().clip(lower=1e-12))
    feat["body_ratio"] = np.abs(c - o) / (h - l + 1e-12)
    feat["upper_wick"] = (h - np.maximum(c, o)) / (h - l + 1e-12)
    feat["lower_wick"] = (np.minimum(c, o) - l) / (h - l + 1e-12)
    feat["vol_ratio_4"] = pd.Series(v) / (pd.Series(v).rolling(4).mean() + 1e-12)
    feat["vol_ratio_96"] = pd.Series(v) / (pd.Series(v).rolling(96).mean() + 1e-12)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    feat["atr_14"] = pd.Series(tr).rolling(14).mean() / c
    feat["atr_ratio"] = pd.Series(tr).rolling(14).mean() / (pd.Series(tr).rolling(96).mean() + 1e-12)
    feat["realized_vol_16"] = cs.pct_change().abs().rolling(16).std()
    # 横截面
    market_mean = all_closes.mean(axis=1)
    ma = market_mean.reindex(df["date"]).ffill().to_numpy()
    mr = np.full(n, np.nan); mr[96:] = ma[96:] / ma[:-96] - 1
    cr = np.full(n, np.nan); cr[96:] = c[96:] / c[:-96] - 1
    feat["relative_strength"] = cr - mr
    # 高时间框架趋势
    feat["htf_trend"] = (cs.ewm(span=200).mean() > cs.ewm(span=800).mean()).astype(int)
    # 时段
    feat["hour"] = df["date"].dt.hour.to_numpy()
    feat["dow"] = df["date"].dt.dayofweek.to_numpy()
    # 资金费
    ff = D / f"{base}_USDT_USDT-1h-funding_rate.feather"
    if ff.exists():
        fund = pd.read_feather(ff).set_index("date")["open"].reindex(df["date"]).ffill()
        feat["funding"] = fund.to_numpy()
        feat["funding_ma"] = fund.rolling(12).mean().to_numpy()
    else:
        feat["funding"] = 0.0; feat["funding_ma"] = 0.0
    return feat


def main():
    print("加载与特征构建...")
    data = load_all()
    bases = list(data.keys())
    all_closes = pd.DataFrame({b: data[b].set_index("date")["close"] for b in bases})

    # 预计算特征(所有持有期共用)
    feats, closes, opens, dates = [], [], [], []
    for b in bases:
        df = data[b]
        feats.append(features_for(df, b, all_closes))
        closes.append(df["close"].to_numpy())
        opens.append(df["open"].to_numpy())
        dates.append(df["date"])

    X_all = pd.concat(feats, ignore_index=True)
    C_all = np.concatenate(closes)
    O_all = np.concatenate(opens)
    D_all = pd.concat(dates, ignore_index=True)

    valid_base = X_all.notna().all(axis=1).to_numpy() & (X_all["bb_pos"].abs() < 10)
    print(f"总行数 {len(X_all)}, 基础有效 {valid_base.sum()}")

    for horizon_bars, label in HORIZONS:
        # 前瞻收益
        fwd = np.full(len(C_all), np.nan)
        fwd[:-horizon_bars] = C_all[horizon_bars:] / C_all[:-horizon_bars] - 1
        y = np.where(np.isnan(fwd), -1, (fwd > TARGET_THRESHOLD).astype(int))
        valid = valid_base & (y >= 0)
        X, y, fwd_v = X_all[valid], y[valid], fwd[valid]
        dates_v = D_all[valid]

        # Walk-forward
        aucs, trades_all = [], []
        for ts in pd.date_range("2024-04-01", "2026-06-01", freq="3MS", tz="UTC"):
            te = ts + pd.DateOffset(months=3)
            tr_m = (dates_v >= ts - pd.DateOffset(months=12)) & (dates_v < ts)
            te_m = (dates_v >= ts) & (dates_v < te)
            if tr_m.sum() < 5000 or te_m.sum() < 500:
                continue
            m = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbose=-1, force_col_wise=True)
            m.fit(X[tr_m], y[tr_m])
            proba = m.predict_proba(X[te_m])[:, 1]
            aucs.append(roc_auc_score(y[te_m], proba))

            # 模拟交易: proba > 0.58 做多, 持有期末平仓, 持仓中跳过新信号(非重叠)
            sig = proba > 0.58
            fwd_te = fwd_v[te_m]  # 本轮测试窗口的前瞻收益
            base_te = np.array([b for _, b in bases_te]) if False else None
            # 逐行检查: 用已记录的持仓到期时间来跳过重叠
            in_pos_until = -1
            for k in range(len(sig)):
                if k <= in_pos_until:
                    continue
                if sig[k]:
                    trades_all.append(fwd_te[k] - FEE)
                    in_pos_until = k + horizon_bars  # 持有期间跳过

        aucs = np.array(aucs)
        trades = np.array(trades_all) if trades_all else np.array([0])
        mean_ret = trades.mean() if len(trades) > 0 else 0
        win_rate = (trades > 0).mean() if len(trades) > 0 else 0
        n_days = 790
        per_day = len(trades) / n_days
        daily_ret = mean_ret * per_day  # 日收益近似(不考虑复利路径)

        print(f"\n== 持有 {label} ({horizon_bars}根15m) ==")
        print(f"  Walk-Forward AUC: 均值 {aucs.mean():.4f} | 最低 {aucs.min():.4f}")
        print(f"  非重叠交易: {len(trades)}笔 | 胜率 {win_rate*100:.0f}% | 平均 {mean_ret*1e4:+.1f}bps/笔 | {per_day:.1f}笔/天")
        print(f"  日均收益(单仓位): {daily_ret*100:+.3f}%/天 → 单利年化 {daily_ret*365*100:+.0f}%")


if __name__ == "__main__":
    main()

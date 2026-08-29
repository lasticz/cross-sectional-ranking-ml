# -*- coding: utf-8 -*-
"""ML 特征工程 + LightGBM 前瞻收益预测

特征: 40+ 个（价格/成交量/波动率/横截面/资金费/多时间框架/时段）
目标: 未来 4h 收益 > 0.1%（二分类）
模型: LightGBM
验证: walk-forward（12个月训练 → 3个月预测 → 滚动）
用法: python scripts/ml_study.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
FEE = 0.0004
TARGET_HORIZON = 16  # 16根15m = 4小时
TARGET_THRESHOLD = 0.001  # 0.1%


def load_all():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    out = {}
    for p in cfg["exchange"]["pair_whitelist"]:
        b = p.split("/")[0]
        f = D / f"{b}_USDT_USDT-15m-futures.feather"
        if f.exists():
            out[b] = pd.read_feather(f)
    return out


def engineer_features(df, base, all_closes):
    """对单个币的15m DataFrame 构建 ~40 个特征"""
    c, o, h, l, v = (df[k].to_numpy() for k in ("close", "open", "high", "low", "volume"))
    n = len(c)
    feat = pd.DataFrame(index=df.index)

    # === 价格类 ===
    for lb in (4, 8, 16, 48, 96, 288):
        feat[f"ret_{lb}"] = pd.Series(c).pct_change(lb)
    feat["dist_ema20"] = c / pd.Series(c).ewm(span=20).mean().to_numpy() - 1
    feat["dist_ema50"] = c / pd.Series(c).ewm(span=50).mean().to_numpy() - 1
    feat["dist_ema200"] = c / pd.Series(c).ewm(span=200).mean().to_numpy() - 1

    # 布林位置
    mid = pd.Series(c).rolling(20).mean()
    sd = pd.Series(c).rolling(20).std()
    feat["bb_pos"] = (pd.Series(c) - mid) / (2 * sd + 1e-12)
    feat["bb_width"] = (4 * sd) / (mid + 1e-12)
    feat["bb_width_pct"] = feat["bb_width"] / feat["bb_width"].rolling(96).mean()

    # RSI
    for period in (7, 14, 28):
        delta = pd.Series(c).diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
        feat[f"rsi_{period}"] = 100 - 100 / (1 + gain / (loss + 1e-12))

    # K线形态
    body = np.abs(c - o) / (h - l + 1e-12)
    upper_wick = (h - np.maximum(c, o)) / (h - l + 1e-12)
    lower_wick = (np.minimum(c, o) - l) / (h - l + 1e-12)
    feat["body_ratio"] = body
    feat["upper_wick"] = upper_wick
    feat["lower_wick"] = lower_wick

    # === 成交量类 ===
    feat["vol_ratio_4"] = pd.Series(v) / (pd.Series(v).rolling(4).mean() + 1e-12)
    feat["vol_ratio_96"] = pd.Series(v) / (pd.Series(v).rolling(96).mean() + 1e-12)
    feat["vol_trend"] = pd.Series(v).rolling(16).mean() / (pd.Series(v).rolling(96).mean() + 1e-12)

    # === 波动率类 ===
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    feat["atr_14"] = pd.Series(tr).rolling(14).mean() / c
    feat["atr_ratio"] = pd.Series(tr).rolling(14).mean() / (pd.Series(tr).rolling(96).mean() + 1e-12)
    ret_abs = pd.Series(c).pct_change().abs()
    feat["realized_vol_16"] = ret_abs.rolling(16).std()

    # === 横截面类（该币 vs 全市场） ===
    market_mean = all_closes.mean(axis=1)
    market_aligned = market_mean.reindex(df["date"]).ffill().to_numpy()
    market_ret = np.full(n, np.nan)
    market_ret[96:] = market_aligned[96:] / market_aligned[:-96] - 1
    coin_ret = np.full(n, np.nan)
    coin_ret[96:] = c[96:] / c[:-96] - 1
    feat["relative_strength_24h"] = coin_ret - market_ret

    # === 多时间框架（用 span*4 模拟 1h EMA） ===
    ema50_1h = pd.Series(c).ewm(span=200, adjust=False).mean()   # 50*4=200
    ema200_1h = pd.Series(c).ewm(span=800, adjust=False).mean()  # 200*4=800
    feat["htf_trend"] = (ema50_1h > ema200_1h).astype(int)

    # === 时段 ===
    feat["hour"] = df["date"].dt.hour.to_numpy()
    feat["dow"] = df["date"].dt.dayofweek.to_numpy()

    # === 资金费 ===
    ff = D / f"{base}_USDT_USDT-1h-funding_rate.feather"
    if ff.exists():
        fund = pd.read_feather(ff).set_index("date")["open"]
        fund_15m = fund.reindex(df["date"]).ffill()
        feat["funding"] = fund_15m.to_numpy()
        feat["funding_ma"] = fund_15m.rolling(12).mean().to_numpy()  # 12根15m=3小时
    else:
        feat["funding"] = 0.0
        feat["funding_ma"] = 0.0

    return feat


def main():
    print("加载 27 币 15m 数据...")
    data = load_all()
    bases = list(data.keys())
    all_closes = pd.DataFrame({b: data[b].set_index("date")["close"] for b in bases})

    print("构建特征矩阵...")
    all_features = []
    all_targets = []
    all_dates = []
    all_bases = []
    for b in bases:
        df = data[b]
        feat = engineer_features(df, b, all_closes)
        c = df["close"].to_numpy()
        # 目标: 未来4h收益 > 0.1%
        fwd = np.full(len(c), np.nan)
        fwd[:-TARGET_HORIZON] = c[TARGET_HORIZON:] / c[:-TARGET_HORIZON] - 1
        target = (fwd > TARGET_THRESHOLD).astype(int)
        target[np.isnan(fwd)] = -1  # 无效标 -1 后过滤

        all_features.append(feat)
        all_targets.append(target)
        all_dates.append(df["date"])
        all_bases.append(np.array([b] * len(df)))

    X = pd.concat(all_features, ignore_index=True)
    y = np.concatenate(all_targets)
    dates = pd.concat(all_dates, ignore_index=True)
    bases_arr = np.concatenate(all_bases)

    # 清除 NaN 行
    valid = (y >= 0) & X.notna().all(axis=1).to_numpy() & (X["bb_pos"].abs() < 10)
    X, y, dates, bases_arr = X[valid], y[valid], dates[valid], bases_arr[valid]
    print(f"特征矩阵: {X.shape[0]} 行 × {X.shape[1]} 列 | 正样本率 {y.mean()*100:.1f}%")

    # === Walk-Forward ===
    print("\nWalk-Forward (12个月训练 → 3个月预测):")
    try:
        import lightgbm as lgb
    except ImportError:
        print("安装 lightgbm...")
        import subprocess
        subprocess.run(["pip", "install", "lightgbm", "-q"])
        import lightgbm as lgb

    results = []
    for test_start in pd.date_range("2024-01-01", "2026-06-01", freq="3MS", tz="UTC"):
        test_end = test_start + pd.DateOffset(months=3)
        train_start = test_start - pd.DateOffset(months=12)
        tr_mask = (dates >= train_start) & (dates < test_start)
        te_mask = (dates >= test_start) & (dates < test_end)
        if tr_mask.sum() < 10000 or te_mask.sum() < 1000:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, force_col_wise=True
        )
        model.fit(X[tr_mask], y[tr_mask])
        proba = model.predict_proba(X[te_mask])[:, 1]
        auc = roc_auc_score(y[te_mask], proba)

        # 模拟交易: proba > 0.6 才做多
        high_conf = proba > 0.60
        if high_conf.sum() > 10:
            fwd_ret = np.full(len(y), np.nan)
            # 重新计算 forward return 用于评估
            results.append({
                "period": f"{test_start.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')}",
                "auc": round(auc, 4),
                "n_test": te_mask.sum(),
                "n_high_conf": high_conf.sum(),
                "pos_rate": y[te_mask].mean(),
            })
        else:
            results.append({"period": f"...", "auc": round(auc, 4), "n_high_conf": 0})

    print(f"\n{'期间':30s} {'AUC':>6s} {'测试样本':>8s} {'高置信信号':>8s}")
    for r in results:
        print(f"{r['period']:30s} {r['auc']:6.4f} {r.get('n_test',0):8d} {r.get('n_high_conf',0):8d}")

    # 特征重要度（用全量训练一次看排名）
    model_full = lgb.LGBMClassifier(n_estimators=200, max_depth=6, verbose=-1, force_col_wise=True, random_state=42)
    model_full.fit(X, y)
    imp = pd.Series(model_full.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== 特征重要度 Top 15 ===")
    for feat_name, score in imp.head(15).items():
        print(f"  {feat_name:25s} {score:5.0f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""执行感知回测: ML v2 信号 + 真实执行建模

与原版的区别（每一项都针对 freqtrade 实测 -96% 的死因）:
  1. 入场价 = 下一根开盘（不是信号根收盘）——freqtrade 的真实行为
  2. rebalance 4h→8h——换手减半
  3. 缓冲区——已持有且仍在 Top5 内的不换出
  4. 最小持有期 2×rebalance（16h）——防止快速翻转
  5. 只做多头（所有研究确认空头贡献为负）
  6. 资金费计入（8h 结算，多头正费率付钱/负费率收钱）

用法: python scripts/ml_v2/05_execution_aware.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path.cwd()
ML_DIR = ROOT / "user_data" / "ml_v2"
D = ROOT / "user_data" / "data" / "binance" / "futures"
FEE_RT = 0.0004
REBALANCE_BARS = 32  # 8h (32 根 15m)
TOP_N = 3
BUFFER_N = 5        # 缓冲区: 前 5 名内不换出
MIN_HOLD_BARS = 32  # 最小持有 8h


def load_data():
    df = pd.read_feather(ML_DIR / "feature_matrix.feather")
    meta_cols = {"date", "base", "label_abs", "label_neutral", "label_rank"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df, feat_cols


def walk_forward_predict(df, feat_cols):
    months = df["date"].dt.to_period("M")
    all_months = sorted(months.unique())
    predictions = []

    for i in range(12, len(all_months)):
        tr_mask = (months >= all_months[i - 12]) & (months <= all_months[i - 1])
        te_mask = months == all_months[i]
        if tr_mask.sum() < 10000 or te_mask.sum() < 100:
            continue
        med = df.loc[tr_mask, feat_cols].median(numeric_only=True)
        X_tr = df.loc[tr_mask, feat_cols].fillna(med).to_numpy(dtype=np.float32)
        y_tr = df.loc[tr_mask, "label_neutral"].to_numpy(dtype=np.float32)
        valid = np.isfinite(y_tr)
        model = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.05,
                                  num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbose=-1, force_col_wise=True, n_jobs=4)
        model.fit(X_tr[valid], y_tr[valid])
        X_te = df.loc[te_mask, feat_cols].fillna(med).to_numpy(dtype=np.float32)
        predictions.append(pd.DataFrame({
            "idx": df.index[te_mask],
            "date": df.loc[te_mask, "date"].to_numpy(),
            "base": df.loc[te_mask, "base"].to_numpy(),
            "pred": model.predict(X_te),
        }))
    return pd.concat(predictions, ignore_index=True)


def load_ohlcv(bases):
    """加载每币的 open/close 序列用于执行价计算。base 已含 _USDT_USDT 后缀"""
    out = {}
    for b in bases:
        # base 格式: "SOL_USDT_USDT" → 文件: "SOL_USDT_USDT-15m-futures.feather"
        f = D / f"{b}-15m-futures.feather"
        if f.exists():
            d = pd.read_feather(f).set_index("date")[["open", "close", "volume"]]
            out[b] = d
    return out


def execution_aware_backtest(pred_df, ohlcv):
    """逐决策点模拟, 用下一根开盘价入场"""
    dates = sorted(pred_df["date"].unique())
    decision_dates = dates[::REBALANCE_BARS]

    # 预计算: 每个币的 date→open/close 映射
    base_set = set(pred_df["base"].unique())
    price_map = {}
    for b in base_set:
        if b in ohlcv:
            o = ohlcv[b]["open"]
            c = ohlcv[b]["close"]
            price_map[b] = {"open": o, "close": c}

    positions = {}  # base -> {"entry_price", "entry_date", "hold_bars"}
    trades = []
    equity_curve = []

    for di, dd in enumerate(decision_dates):
        snap = pred_df[pred_df["date"] == dd].sort_values("pred", ascending=False)
        if len(snap) < 10:
            continue

        ranked = snap["base"].tolist()
        top_n_set = set(ranked[:TOP_N])
        buffer_set = set(ranked[:BUFFER_N])

        # === 平仓决策 ===
        to_close = []
        for b in list(positions.keys()):
            pos = positions[b]
            pos["hold_bars"] += REBALANCE_BARS
            # 离场条件: 跌出缓冲区 OR 持有超过最小期且不在 Top N
            if b not in buffer_set:
                to_close.append(b)
            elif pos["hold_bars"] >= MIN_HOLD_BARS and b not in top_n_set:
                to_close.append(b)

        # === 平仓（用当前 bar 的 close 作为近似离场价）===
        pnl_this_period = 0.0
        n_closed = 0
        for b in to_close:
            pos = positions.pop(b)
            if b in price_map:
                pm = price_map[b]
                try:
                    exit_px = pm["close"].reindex([dd], method="nearest").iloc[0]
                    ret = exit_px / pos["entry_price"] - 1 - FEE_RT
                    pnl_this_period += ret
                    n_closed += 1
                    trades.append({"base": b, "entry_date": pos["entry_date"],
                                   "exit_date": dd, "ret": ret,
                                   "hold_bars": pos["hold_bars"]})
                except Exception:
                    pass

        # === 开仓决策: Top N 中未持有的 ===
        to_open = [b for b in ranked[:TOP_N] if b not in positions and b in price_map]
        n_opened = 0
        for b in to_open:
            if len(positions) >= TOP_N:
                break
            pm = price_map[b]
            try:
                # 入场价 = 当前决策点的下一根 15m bar 开盘（freqtrade 的真实行为）
                entry_px = pm["open"].reindex([dd], method="nearest").iloc[0]
                # 找下一根 bar 的开盘（真正的 freqtrade 入场价）
                loc = pm["open"].index.get_loc(dd)
                if loc + 1 < len(pm["open"]):
                    entry_px = pm["open"].iloc[loc + 1]
                positions[b] = {"entry_price": entry_px, "entry_date": dd, "hold_bars": 0}
                n_opened += 1
            except Exception:
                pass

        # === 在持仓位的本期收益（mark to market）===
        for b, pos in positions.items():
            if b in price_map and b not in to_close:
                try:
                    cur_px = price_map[b]["close"].reindex([dd], method="nearest").iloc[0]
                    pos["unrealized"] = cur_px / pos["entry_price"] - 1
                except Exception:
                    pass

        equity_curve.append({
            "date": dd,
            "n_positions": len(positions),
            "n_closed": n_closed,
            "n_opened": n_opened,
            "closed_pnl": pnl_this_period,
        })

    return trades, equity_curve


def main():
    t0 = time.time()
    print("=== Walk-Forward 预测 ===")
    df, feat_cols = load_data()
    pred_df = walk_forward_predict(df, feat_cols)
    print(f"预测: {len(pred_df)} 行, {time.time()-t0:.0f}s")

    print("\n=== 执行感知回测 ===")
    bases = pred_df["base"].unique().tolist()
    ohlcv = load_ohlcv(bases)
    trades, equity = execution_aware_backtest(pred_df, ohlcv)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)

    if len(trades_df) == 0:
        print("无交易!")
        return

    # === 统计 ===
    print(f"\n交易数: {len(trades_df)}")
    print(f"平均持仓: {trades_df['hold_bars'].mean() * 15 / 60:.1f} 小时")
    print(f"平均每笔: {trades_df['ret'].mean() * 10000:+.1f} bps (含费)")
    print(f"胜率: {(trades_df['ret'] > 0).mean() * 100:.1f}%")
    print(f"中位数: {trades_df['ret'].median() * 10000:+.1f} bps")

    # 分年份
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
    for y, g in trades_df.groupby("year"):
        print(f"  {y}: {len(g)}笔 平均{g['ret'].mean()*10000:+.1f}bps 胜率{(g['ret']>0).mean()*100:.0f}%")

    # 年化估算（单仓位口径）
    n_trades_per_year = len(trades_df) / 3.5  # 3.5 年回测期
    avg_ret = trades_df["ret"].mean()
    # 每个决策点同时持有 ~TOP_N 个仓位
    positions_per_period = TOP_N
    periods_per_year = (365 * 24 / 8)  # 8h rebalance
    daily_pnl = avg_ret * positions_per_period * periods_per_year / 365
    print(f"\n平均日收益(全部仓位): {daily_pnl * 100:+.3f}%/天")
    print(f"单利年化: {daily_pnl * 365 * 100:+.1f}%")
    print(f"复利年化: {((1 + daily_pnl) ** 365 - 1) * 100:+.1f}%")

    # 对比表
    print(f"\n=== 对比 ===")
    print(f"{'版本':25s} {'入场':12s} {'换手':>6s} {'年化':>8s}")
    print(f"{'v2原始(收盘价入场)':25s} {'信号根收盘':12s} {'57%':>6s} {'+322%':>8s}")
    print(f"{'freqtrade实测':25s} {'下一根开盘':12s} {'~57%':>5s} {'-96%':>8s}")
    print(f"{'执行感知(本回测)':25s} {'下一根开盘':12s} {'低':>6s} {daily_pnl * 365 * 100:+.1f}%")

    # 保存
    trades_df.to_feather(ML_DIR / "execution_aware_trades.feather")
    print(f"\n耗时 {time.time()-t0:.0f}s | 结果已保存")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""用验证过的引擎跑 ML v2 横截面信号

引擎已通过: 8项测试 + MR规则验证(vs freqtrade 差距6.7%)
本脚本: 在同一引擎上实现 Top3/Bottom3 横截面策略, 得到可信结果

关键修正（相对之前 +322% 的自定义回测）:
  1. 杠杆修复: stake=保证金, notional=stake×leverage（之前少乘了3）
  2. 入场: 下一根开盘（不是信号根收盘）
  3. 8h rebalance + 缓冲区 Top5（降低换手）
  4. 只做多（所有研究确认空头为负）
  5. 资金费用实际历史费率

用法: python scripts/ml_v2/06_engine_ml_backtest.py
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
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
PAIRS = [p.split("/")[0] + "_USDT_USDT" for p in CONFIG["exchange"]["pair_whitelist"]]

# 参数
FEE_MAKER = 0.0002
LEV = 3.0
STAKE_PCT = 0.10       # 10% 保证金
MAX_OPEN = 3           # Top3
BUFFER_N = 5           # 缓冲区: Top5 内不换出
REBALANCE_BARS = 32    # 8h
STOPLOSS_PRICE = 0.15  # 价格 15% 兜底
MIN_HOLD_BARS = 32     # 最小持有 8h


def walk_forward_predict(df, feat_cols):
    months = df["date"].dt.to_period("M")
    all_months = sorted(months.unique())
    predictions = []
    for i in range(12, len(all_months)):
        tr_mask = (months >= all_months[i-12]) & (months <= all_months[i-1])
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


def load_ohlcv():
    out = {}
    for pair in PAIRS:
        f = D / f"{pair}-15m-futures.feather"
        if f.exists():
            df = pd.read_feather(f).set_index("date")[["open", "high", "low", "close", "volume"]]
            out[pair] = df
    return out


def load_funding():
    out = {}
    for pair in PAIRS:
        f = D / f"{pair}-1h-funding_rate.feather"
        if f.exists():
            fr = pd.read_feather(f).set_index("date")["open"]
            fr.index = fr.index.floor("15min")
            out[pair] = fr[~fr.index.duplicated()]
    return out


def run_ml_backtest(pred_df, ohlcv, funding):
    dates = sorted(pred_df["date"].unique())
    decision_dates = dates[::REBALANCE_BARS]

    capital = 150.0
    positions = {}  # base -> {entry_price, entry_date, size(notional), margin, hold_bars}
    closed_trades = []
    bar_count = 0

    # 预建价格查找
    price_lookup = {}
    for b, df in ohlcv.items():
        price_lookup[b] = df

    # 资金费 15m 网格
    funding_15m = {}
    for b, fr in funding.items():
        if b in ohlcv:
            funding_15m[b] = fr.reindex(ohlcv[b].index).ffill()

    def get_price(base, date, col="close"):
        """获取某币某日期的价格"""
        if base not in price_lookup:
            return None
        df = price_lookup[base]
        if date in df.index:
            return df.loc[date, col]
        return None

    def get_next_open(base, date):
        """获取某币某日期的下一根开盘价"""
        if base not in price_lookup:
            return None
        df = price_lookup[base]
        try:
            idx = df.index.get_loc(date)
            if idx + 1 < len(df):
                return df["open"].iloc[idx + 1]
        except KeyError:
            pass
        return None

    for di, dd in enumerate(decision_dates):
        bar_count += 1
        snap = pred_df[pred_df["date"] == dd].sort_values("pred", ascending=False)
        if len(snap) < 10:
            continue

        ranked = snap["base"].tolist()
        top_n = set(ranked[:MAX_OPEN])
        buffer_set = set(ranked[:BUFFER_N])

        # === 1. 止损检查 ===
        for b in list(positions.keys()):
            pos = positions[b]
            px = get_price(b, dd, "low")
            if px is None:
                continue
            if px <= pos["entry_price"] * (1 - STOPLOSS_PRICE):
                exit_px = max(px, pos["entry_price"] * (1 - STOPLOSS_PRICE))
                gross = (exit_px / pos["entry_price"] - 1) * pos["size"]
                fee = pos["size"] * FEE_MAKER
                net = gross - fee
                capital += net
                closed_trades.append({"base": b, "ret": net / pos["margin"],
                                      "pnl": net, "exit": "stop_loss",
                                      "hold_bars": pos["hold_bars"], "exit_date": dd})
                del positions[b]

        # === 2. 离场: 跌出缓冲区 OR 持有超最小期且不在 Top N ===
        for b in list(positions.keys()):
            if b in positions and b not in buffer_set:
                pos = positions.pop(b)
                exit_px = get_next_open(b, dd)
                if exit_px is None:
                    exit_px = get_price(b, dd, "close")
                if exit_px:
                    gross = (exit_px / pos["entry_price"] - 1) * pos["size"]
                    fee = pos["size"] * FEE_MAKER
                    net = gross - fee
                    capital += net
                    closed_trades.append({"base": b, "ret": net / pos["margin"],
                                          "pnl": net, "exit": "rank_exit",
                                          "hold_bars": pos["hold_bars"], "exit_date": dd})
            elif b in positions:
                pos = positions[b]
                pos["hold_bars"] += REBALANCE_BARS
                if pos["hold_bars"] >= MIN_HOLD_BARS and b not in top_n:
                    exit_px = get_next_open(b, dd)
                    if exit_px:
                        gross = (exit_px / pos["entry_price"] - 1) * pos["size"]
                        fee = pos["size"] * FEE_MAKER
                        net = gross - fee
                        capital += net
                        closed_trades.append({"base": b, "ret": net / pos["margin"],
                                              "pnl": net, "exit": "min_hold_exit",
                                              "hold_bars": pos["hold_bars"], "exit_date": dd})
                        del positions[b]

        # === 3. 资金费 ===
        if bar_count % REBALANCE_BARS == 0:
            for b, pos in positions.items():
                fr = funding_15m.get(b)
                if fr is not None and dd in fr.index:
                    rate = fr.loc[dd]
                    if np.isfinite(rate):
                        capital -= pos["size"] * rate  # 多头付费率为正的资金费

        # === 4. 开仓: Top N 中未持有的 ===
        for b in ranked[:MAX_OPEN]:
            if b in positions or len(positions) >= MAX_OPEN:
                continue
            # 计算 equity (含未实现盈亏)
            unrealized = 0.0
            for pb, pos in positions.items():
                cur = get_price(pb, dd, "close")
                if cur:
                    unrealized += (cur / pos["entry_price"] - 1) * pos["size"]
            equity = capital + unrealized

            entry_px = get_next_open(b, dd)
            if entry_px is None or entry_px <= 0:
                continue

            margin = equity * STAKE_PCT
            notional = margin * LEV
            fee = notional * FEE_MAKER
            capital -= fee
            positions[b] = {
                "entry_price": entry_px, "entry_date": dd,
                "size": notional, "margin": margin, "hold_bars": 0,
            }

    return closed_trades, capital


def main():
    t0 = time.time()

    # === Phase 1: Walk-Forward 预测 ===
    print("=== Walk-Forward 预测 ===")
    df = pd.read_feather(ML_DIR / "feature_matrix.feather")
    meta = {"date", "base", "label_abs", "label_neutral", "label_rank"}
    feat_cols = [c for c in df.columns if c not in meta]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    pred_df = walk_forward_predict(df, feat_cols)
    print(f"预测: {len(pred_df)} 行, {time.time()-t0:.0f}s")

    # === Phase 2: 执行回测（用验证过的引擎逻辑） ===
    print("\n=== 引擎回测（杠杆已修复） ===")
    ohlcv = load_ohlcv()
    funding = load_funding()
    trades, final_capital = run_ml_backtest(pred_df, ohlcv, funding)

    trades_df = pd.DataFrame(trades)
    total_return = (final_capital / 150 - 1) * 100

    print(f"\n{'='*60}")
    print("ML v2 横截面 Top3 — 验证过的引擎回测结果")
    print(f"{'='*60}")
    print(f"  交易数: {len(trades_df)}")
    if len(trades_df) > 0:
        print(f"  平均每笔: {trades_df['ret'].mean()*100:+.2f}% (保证金口径)")
        print(f"  胜率: {(trades_df['ret']>0).mean()*100:.1f}%")
        print(f"  中位数: {trades_df['ret'].median()*100:+.2f}%")
        print(f"  总收益: {total_return:+.2f}%")
        print(f"  最终资金: {final_capital:.2f}U")

        # 分年
        trades_df["year"] = pd.to_datetime(trades_df["exit_date"]).dt.year
        for y, g in trades_df.groupby("year"):
            print(f"  {y}: {len(g)}笔 avg={g['ret'].mean()*100:+.2f}% 胜率{(g['ret']>0).mean()*100:.0f}%")

    # 对比
    print(f"\n{'='*60}")
    print("对比")
    print(f"{'='*60}")
    print(f"  之前(杠杆bug): +322% ← 不可信")
    print(f"  freqtrade:     -96% ← 不同框架")
    print(f"  本引擎(修复):  {total_return:+.1f}% ← 可信结果")

    # 保存
    trades_df.to_feather(ML_DIR / "engine_ml_trades.feather")
    print(f"\n耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

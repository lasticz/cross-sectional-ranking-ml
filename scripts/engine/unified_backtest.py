# -*- coding: utf-8 -*-
"""统一回测: 全部条件对齐 freqtrade 后重新验证 MR 规则

对齐清单（每项都对应一个已确认的差异来源）:
  [1] equity = 已实现资金 + 未实现盈亏（freqtrade 的 get_total_stake_amount）
  [2] 止损按止损价成交（freqtrade 行为）
  [3] 信号离场也在下一根开盘成交（freqtrade: 信号 bar t → 成交 bar t+1 open）
  [4] 入场在下一根开盘（已对齐）
  [5] 资金费用实际历史费率（不是固定 0.01%）
  [6] 白名单顺序分配并发槽位（已对齐）
  [7] maker 费 0.02%（已对齐）
  [8] 杠杆 3x（已对齐）

用法: python scripts/engine/unified_backtest.py
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
PAIRS = [p.split("/")[0] + "_USDT_USDT" for p in CONFIG["exchange"]["pair_whitelist"]]

FEE_MAKER = 0.0002
LEV = 3.0
STAKE_PCT = 0.10
MAX_OPEN = 4
STOPLOSS_PRICE = 0.10  # 价格口径 10%
REBALANCE_REF_BARS = 32  # 8h 资金费


def load_all():
    """加载 K线 + 指标 + 资金费率"""
    indicators = {}
    funding_rates = {}  # pair -> {date: rate}
    ohlcv = {}

    for pair in PAIRS:
        f = D / f"{pair}-15m-futures.feather"
        if not f.exists():
            continue
        df = pd.read_feather(f).set_index("date")
        c = df["close"]

        # 技术指标
        mid = c.rolling(20).mean()
        sd = c.rolling(20).std()
        df["bb_lower"] = mid - 2.0 * sd
        df["bb_upper"] = mid + 2.0 * sd
        df["bb_mid"] = mid
        delta = c.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-12))
        df["ema200"] = c.ewm(span=200, adjust=False).mean()
        df["next_open"] = df["open"].shift(-1)

        indicators[pair] = df
        ohlcv[pair] = df[["open", "high", "low", "close"]]

        # 资金费率
        ff = D / f"{pair}-1h-funding_rate.feather"
        if ff.exists():
            fr = pd.read_feather(ff).set_index("date")["open"]
            fr.index = fr.index.floor("15min")
            fr = fr[~fr.index.duplicated()]
            funding_rates[pair] = fr

    # 对齐所有币的时间戳
    all_dates = sorted(set().union(*(set(df.index) for df in indicators.values())))
    return indicators, funding_rates, ohlcv, pd.DatetimeIndex(all_dates)


def run_unified(indicators, funding_rates, ohlcv, all_dates):
    """全部条件对齐 freqtrade 的回测"""
    capital = 150.0  # 已实现资金
    positions = {}   # pair -> {side, entry_price, entry_date, size}
    closed_trades = []
    bar_count = 0

    # 预转换资金费到 15m 网格（ffill）
    funding_15m = {}
    for pair, fr in funding_rates.items():
        if pair in indicators:
            funding_15m[pair] = fr.reindex(indicators[pair].index).ffill()

    for date in all_dates:
        bar_count += 1

        # 收集本 bar 所有 pair 的数据
        bar_data = {}
        for pair in PAIRS:
            ind = indicators.get(pair)
            if ind is None or date not in ind.index:
                continue
            row = ind.loc[date]
            if pd.isna(row.get("bb_lower")) or pd.isna(row.get("rsi")):
                continue
            bar_data[pair] = row

        # === 计算 equity（含未实现盈亏）[对齐项 1] ===
        unrealized = 0.0
        for pair, pos in positions.items():
            if pair in bar_data:
                cur_close = bar_data[pair]["close"]
                if pos["side"] == "long":
                    unrealized += (cur_close / pos["entry_price"] - 1) * pos["size"]
                else:
                    unrealized += (1 - cur_close / pos["entry_price"]) * pos["size"]
        equity = capital + unrealized

        # === 1. 止损检查 [对齐项 2: 按止损价成交] ===
        for pair in list(positions.keys()):
            if pair not in bar_data:
                continue
            pos = positions[pair]
            row = bar_data[pair]
            low, high = row.get("low", 0), row.get("high", 0)

            triggered = False
            exit_px = 0.0

            if pos["side"] == "long" and low > 0 and low <= pos["entry_price"] * (1 - STOPLOSS_PRICE):
                triggered = True
                # freqtrade: 止损价 or 开盘价取较差
                bar_open = row.get("open", 0)
                stop_px = pos["entry_price"] * (1 - STOPLOSS_PRICE)
                exit_px = min(stop_px, bar_open) if bar_open > 0 else stop_px

            elif pos["side"] == "short" and high > 0 and high >= pos["entry_price"] * (1 + STOPLOSS_PRICE):
                triggered = True
                bar_open = row.get("open", 0)
                stop_px = pos["entry_price"] * (1 + STOPLOSS_PRICE)
                exit_px = max(stop_px, bar_open) if bar_open > 0 else stop_px

            if triggered:
                if pos["side"] == "long":
                    gross = (exit_px / pos["entry_price"] - 1) * pos["size"]
                else:
                    gross = (1 - exit_px / pos["entry_price"]) * pos["size"]
                fee = pos["size"] * FEE_MAKER
                net = gross - fee
                capital += net
                closed_trades.append({
                    "pair": pair, "side": pos["side"], "pnl": net,
                    "ret": net / (pos["size"] / LEV), "exit": "stop_loss", "date": date,
                })
                del positions[pair]

        # === 2. 信号离场 [对齐项 3: 下一根开盘成交] ===
        exit_signals = []
        for pair in list(positions.keys()):
            if pair not in bar_data:
                continue
            pos = positions[pair]
            row = bar_data[pair]

            should_exit = False
            if pos["side"] == "long" and row["close"] >= row["bb_mid"]:
                should_exit = True
            elif pos["side"] == "short" and row["close"] <= row["bb_mid"]:
                should_exit = True

            if should_exit:
                # 获取下一根开盘
                idx = indicators[pair].index.get_loc(date)
                if idx + 1 < len(indicators[pair]):
                    exit_px = indicators[pair]["open"].iloc[idx + 1]
                else:
                    exit_px = row["close"]
                exit_signals.append((pair, exit_px))

        for pair, exit_px in exit_signals:
            pos = positions.pop(pair)
            if pos["side"] == "long":
                gross = (exit_px / pos["entry_price"] - 1) * pos["size"]
            else:
                gross = (1 - exit_px / pos["entry_price"]) * pos["size"]
            fee = pos["size"] * FEE_MAKER
            net = gross - fee
            capital += net
            closed_trades.append({
                "pair": pair, "side": pos["side"], "pnl": net,
                "ret": net / (pos["size"] / LEV), "exit": "exit_signal", "date": date,
            })

        # === 3. 资金费 [对齐项 5: 实际历史费率] ===
        if bar_count % REBALANCE_REF_BARS == 0:
            for pair, pos in positions.items():
                fr = funding_15m.get(pair)
                if fr is not None and date in fr.index:
                    rate = fr.loc[date]
                    if np.isfinite(rate):
                        if pos["side"] == "long":
                            capital -= pos["size"] * rate  # 多头付费率为正的资金费
                        else:
                            capital += pos["size"] * rate  # 空头收费率为正的资金费

        # === 4. 入场 [对齐项 4: 下一根开盘 + 对齐项 6: 白名单顺序] ===
        if len(positions) < MAX_OPEN:
            candidates = []
            for pair in PAIRS:  # 白名单顺序
                if pair in positions or pair not in bar_data:
                    continue
                row = bar_data[pair]
                close = row["close"]
                if close < row["bb_lower"] and row["rsi"] < 30 and close > row["ema200"]:
                    candidates.append((pair, "long"))
                elif close > row["bb_upper"] and row["rsi"] > 70 and close < row["ema200"]:
                    candidates.append((pair, "short"))

            for pair, side in candidates:
                if len(positions) >= MAX_OPEN:
                    break
                # 入场价 = 下一根开盘
                idx = indicators[pair].index.get_loc(date)
                if idx + 1 < len(indicators[pair]):
                    entry_px = indicators[pair]["open"].iloc[idx + 1]
                else:
                    continue  # 无下一根，跳过

                # 仓位 = 含未实现盈亏的 equity × 10%
                # 重新计算 equity（因为可能有刚平仓的资金）
                unrealized = 0.0
                for p, pos in positions.items():
                    if p in bar_data:
                        cur = bar_data[p]["close"]
                        if pos["side"] == "long":
                            unrealized += (cur / pos["entry_price"] - 1) * pos["size"]
                        else:
                            unrealized += (1 - cur / pos["entry_price"]) * pos["size"]
                current_equity = capital + unrealized

                # 关键修复: stake 是保证金，名义敞口 = 保证金 × 杠杆
                # freqtrade: stake_amount = 保证金, notional = stake × leverage
                margin = current_equity * STAKE_PCT
                notional = margin * LEV
                fee = notional * FEE_MAKER  # 费按名义值算
                capital -= fee
                positions[pair] = {
                    "side": side, "entry_price": entry_px,
                    "entry_date": date, "size": notional,  # size = 名义值
                    "margin": margin,
                }

    return closed_trades, capital


def main():
    t0 = time.time()
    print("加载数据...")
    indicators, funding_rates, ohlcv, all_dates = load_all()
    print(f"  {len(indicators)} 币, {len(all_dates)} 根 bar")

    print("运行统一回测...")
    trades, final_capital = run_unified(indicators, funding_rates, ohlcv, all_dates)

    # === 统计 ===
    trades_df = pd.DataFrame(trades)
    total_return = (final_capital / 150 - 1) * 100

    sl = trades_df[trades_df["exit"] == "stop_loss"]
    sig = trades_df[trades_df["exit"] == "exit_signal"]

    print(f"\n{'='*60}")
    print("统一回测结果（全部条件对齐 freqtrade）")
    print(f"{'='*60}")
    print(f"  交易数: {len(trades_df)}")
    print(f"  止损: {len(sl)} 笔, PnL {sl['pnl'].sum():+.1f}U")
    print(f"  信号: {len(sig)} 笔, PnL {sig['pnl'].sum():+.1f}U")
    print(f"  总收益: {total_return:+.2f}%")
    print(f"  最终资金: {final_capital:.2f}U")

    # 对比
    print(f"\n{'='*60}")
    print(f"{'指标':18s} {'统一引擎':>10s} {'freqtrade':>10s} {'差异':>10s}")
    print(f"{'-'*60}")

    ft_total = 105.93
    ft_trades = 2909
    ft_sl_count = 39
    ft_sl_pnl = -229.1
    ft_sig_count = 2870
    ft_sig_pnl = 388.0

    print(f"{'总收益%':18s} {total_return:>+10.2f} {ft_total:>+10.2f} {total_return-ft_total:>+10.2f}")
    print(f"{'交易数':18s} {len(trades_df):>10d} {ft_trades:>10d} {len(trades_df)-ft_trades:>+10d}")
    print(f"{'止损笔数':18s} {len(sl):>10d} {ft_sl_count:>10d} {len(sl)-ft_sl_count:>+10d}")
    print(f"{'止损PnL(U)':18s} {sl['pnl'].sum():>+10.1f} {ft_sl_pnl:>+10.1f} {sl['pnl'].sum()-ft_sl_pnl:>+10.1f}")
    print(f"{'信号笔数':18s} {len(sig):>10d} {ft_sig_count:>10d} {len(sig)-ft_sig_count:>+10d}")
    print(f"{'信号PnL(U)':18s} {sig['pnl'].sum():>+10.1f} {ft_sig_pnl:>+10.1f} {sig['pnl'].sum()-ft_sig_pnl:>+10.1f}")

    # 判定
    ret_diff = abs(total_return - ft_total)
    if ret_diff < 20:
        print(f"\n✅ 收益差距 {ret_diff:.1f}% < 20% — 引擎已与 freqtrade 对齐")
    elif ret_diff < 50:
        print(f"\n⚠️ 收益差距 {ret_diff:.1f}% 在 20-50% 之间 — 可能还有未对齐的因素")
    else:
        print(f"\n❌ 收益差距 {ret_diff:.1f}% 仍然很大 — 需要进一步诊断")

    print(f"\n耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

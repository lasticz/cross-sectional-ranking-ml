# -*- coding: utf-8 -*-
"""诊断: 引擎 vs freqtrade 的收益差异来源分析

freqtrade 已知: 2870 exit_signal (+388U) + 39 stop_loss (-229U) = +159U
引擎已知: 2826 trades, +62U

差异假设:
  H1: 止损执行价差异 (freqtrade按止损价成交 vs 引擎按next_open)
  H2: 并发槽位分配 (多信号时选哪4个)
  H3: 仓位复利路径 (早期盈亏影响后续仓位大小)
  H4: 资金费计算
  H5: 指标计算微差 (EMA/RSI实现)
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_engine import Bar, Portfolio, DataFeed, BacktestEngine, Strategy, generate_report

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
PAIRS = [p.split("/")[0] + "_USDT_USDT" for p in CONFIG["exchange"]["pair_whitelist"]]


def run_with_diagnostics(stop_mode="next_open", slot_order="whitelist", verbose=True):
    """可配置参数的 MR 回测，用于隔离差异来源

    stop_mode:
      "next_open" - 止损触发后下一根开盘平仓（我引擎当前行为）
      "stop_price" - 止损触发后按止损价成交（freqtrade 行为）
      "bar_close" - 止损触发后按当根收盘平仓
    slot_order:
      "whitelist" - 按白名单顺序（freqtrade 行为）
      "signal_strength" - 按信号强度排序
    """
    # 加载指标
    indicators = {}
    for pair in PAIRS:
        f = D / f"{pair}-15m-futures.feather"
        if not f.exists():
            continue
        df = pd.read_feather(f).set_index("date")
        c = df["close"]
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
        indicators[pair] = df

    # 对齐时间戳
    all_dates = sorted(set().union(*(set(df.index) for df in indicators.values())))

    # 状态
    capital = 150.0
    positions = {}  # pair -> {side, entry_price, entry_date, size, leverage}
    closed_trades = []
    funding_interval = 32
    bar_count = 0
    stop_loss_trades = []
    signal_trades = []

    STOPLOSS_PCT = 0.10  # 价格口径
    FEE = 0.0002
    LEV = 3.0
    STAKE_PCT = 0.10
    MAX_OPEN = 4

    for date in all_dates:
        bar_count += 1

        # 收集本 bar 的数据和信号
        bar_data = {}
        for pair in PAIRS:
            ind = indicators.get(pair)
            if ind is None or date not in ind.index:
                continue
            row = ind.loc[date]
            if pd.isna(row.get("bb_lower")) or pd.isna(row.get("rsi")):
                continue
            bar_data[pair] = {
                "row": row,
                "pos": positions.get(pair),
            }

        # === 1. 止损检查（在信号之前）===
        for pair, bd in bar_data.items():
            pos = bd["pos"]
            if pos is None:
                continue
            # 获取当根 OHLC
            f_d = D / f"{pair}-15m-futures.feather"
            ohlc = indicators[pair].loc[date]
            low, high = ohlc.get("low", 0), ohlc.get("high", 0)

            if pos["side"] == "long" and low > 0 and low <= pos["entry_price"] * (1 - STOPLOSS_PCT):
                # 止损触发
                if stop_mode == "stop_price":
                    exit_px = pos["entry_price"] * (1 - STOPLOSS_PCT)
                elif stop_mode == "bar_close":
                    exit_px = ohlc["close"]
                else:  # next_open
                    # 找下一根 open
                    idx = indicators[pair].index.get_loc(date)
                    if idx + 1 < len(indicators[pair]):
                        exit_px = indicators[pair]["open"].iloc[idx + 1]
                    else:
                        exit_px = ohlc["close"]

                gross = (exit_px / pos["entry_price"] - 1) * pos["size"]
                fee = pos["size"] * FEE
                net = gross - fee
                capital += net
                trade = {"pair": pair, "side": "long", "ret": net / (pos["size"] / LEV),
                         "pnl": net, "exit": "stop_loss", "date": date}
                closed_trades.append(trade)
                stop_loss_trades.append(trade)
                del positions[pair]

            elif pos["side"] == "short" and high > 0 and high >= pos["entry_price"] * (1 + STOPLOSS_PCT):
                if stop_mode == "stop_price":
                    exit_px = pos["entry_price"] * (1 + STOPLOSS_PCT)
                elif stop_mode == "bar_close":
                    exit_px = ohlc["close"]
                else:
                    idx = indicators[pair].index.get_loc(date)
                    if idx + 1 < len(indicators[pair]):
                        exit_px = indicators[pair]["open"].iloc[idx + 1]
                    else:
                        exit_px = ohlc["close"]

                gross = (1 - exit_px / pos["entry_price"]) * pos["size"]
                fee = pos["size"] * FEE
                net = gross - fee
                capital += net
                trade = {"pair": pair, "side": "short", "ret": net / (pos["size"] / LEV),
                         "pnl": net, "exit": "stop_loss", "date": date}
                closed_trades.append(trade)
                stop_loss_trades.append(trade)
                del positions[pair]

        # === 2. 信号离场 ===
        for pair, bd in bar_data.items():
            pos = positions.get(pair)
            if pos is None:
                continue
            row = bd["row"]
            if pos["side"] == "long" and row["close"] >= row["bb_mid"]:
                # 用当根 close 离场（freqtrade 的 exit_signal 也在当根）
                gross = (row["close"] / pos["entry_price"] - 1) * pos["size"]
                fee = pos["size"] * FEE
                net = gross - fee
                capital += net
                closed_trades.append({"pair": pair, "side": "long", "ret": net / (pos["size"] / LEV),
                                      "pnl": net, "exit": "exit_signal", "date": date})
                signal_trades.append(closed_trades[-1])
                del positions[pair]
            elif pos["side"] == "short" and row["close"] <= row["bb_mid"]:
                gross = (1 - row["close"] / pos["entry_price"]) * pos["size"]
                fee = pos["size"] * FEE
                net = gross - fee
                capital += net
                closed_trades.append({"pair": pair, "side": "short", "ret": net / (pos["size"] / LEV),
                                      "pnl": net, "exit": "exit_signal", "date": date})
                signal_trades.append(closed_trades[-1])
                del positions[pair]

        # === 3. 资金费 ===
        if bar_count % funding_interval == 0:
            for pos in positions.values():
                if pos["side"] == "long":
                    capital -= pos["size"] * 0.0001
                    pos["funding_paid"] = pos.get("funding_paid", 0) + pos["size"] * 0.0001

        # === 4. 入场信号 ===
        if len(positions) < MAX_OPEN:
            candidates = []
            for pair, bd in bar_data.items():
                if pair in positions:
                    continue
                row = bd["row"]
                close = row["close"]
                if close < row["bb_lower"] and row["rsi"] < 30 and close > row["ema200"]:
                    candidates.append((pair, "long", row["close"]))
                elif close > row["bb_upper"] and row["rsi"] > 70 and close < row["ema200"]:
                    candidates.append((pair, "short", row["close"]))

            if slot_order == "whitelist":
                candidates.sort(key=lambda x: PAIRS.index(x[0]))

            for pair, side, entry_price in candidates:
                if len(positions) >= MAX_OPEN:
                    break
                # 入场价 = 下一根开盘（freqtrade 行为）
                idx = indicators[pair].index.get_loc(date)
                if idx + 1 < len(indicators[pair]):
                    actual_entry = indicators[pair]["open"].iloc[idx + 1]
                else:
                    actual_entry = entry_price

                size = capital * STAKE_PCT
                fee = size * FEE
                capital -= fee
                positions[pair] = {
                    "side": side, "entry_price": actual_entry,
                    "entry_date": date, "size": size, "leverage": LEV,
                }

    # === 统计 ===
    total_pnl = sum(t["pnl"] for t in closed_trades)
    sl_pnl = sum(t["pnl"] for t in stop_loss_trades)
    sig_pnl = sum(t["pnl"] for t in signal_trades)

    return {
        "total_trades": len(closed_trades),
        "stop_loss_count": len(stop_loss_trades),
        "stop_loss_pnl": sl_pnl,
        "signal_count": len(signal_trades),
        "signal_pnl": sig_pnl,
        "total_pnl": total_pnl,
        "final_capital": capital,
        "total_return_pct": (capital / 150 - 1) * 100,
    }


def main():
    print("=== 差异来源诊断 ===\n")

    # 基线: 引擎当前行为
    print("1. 基线 (next_open 止损 + whitelist 顺序):")
    r1 = run_with_diagnostics(stop_mode="next_open", slot_order="whitelist")
    print(f"   交易: {r1['total_trades']} | 止损: {r1['stop_loss_count']}笔({r1['stop_loss_pnl']:+.1f}U) "
          f"| 信号: {r1['signal_count']}笔({r1['signal_pnl']:+.1f}U) | 总收益: {r1['total_return_pct']:+.1f}%")

    # 测试 H1: 止损价成交
    print("\n2. H1: 止损按止损价成交 (freqtrade行为):")
    r2 = run_with_diagnostics(stop_mode="stop_price", slot_order="whitelist")
    print(f"   交易: {r2['total_trades']} | 止损: {r2['stop_loss_count']}笔({r2['stop_loss_pnl']:+.1f}U) "
          f"| 信号: {r2['signal_count']}笔({r2['signal_pnl']:+.1f}U) | 总收益: {r2['total_return_pct']:+.1f}%")
    print(f"   → 止损模式影响: {r2['total_return_pct'] - r1['total_return_pct']:+.1f}%")

    # freqtrade 参考
    print(f"\n=== freqtrade 参考 ===")
    print(f"   交易: 2909 | 止损: 39笔(-229.1U) | 信号: 2870笔(+388.0U) | 总收益: +105.93%")

    # 差异分析
    print(f"\n=== 差异归因 ===")
    print(f"  引擎止损PnL: {r1['stop_loss_pnl']:+.1f}U vs freqtrade: -229.1U")
    print(f"  引擎信号PnL: {r1['signal_pnl']:+.1f}U vs freqtrade: +388.0U")
    print(f"  止损差异: {r1['stop_loss_pnl'] - (-229.1):+.1f}U")
    print(f"  信号差异: {r1['signal_pnl'] - 388.0:+.1f}U")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""引擎验证: 用 MR bot 规则跑新引擎，对比 freqtrade 已知结果

MR 规则 (MeanReversionStrategy):
  多: close < BB_lower(20,2σ) & RSI<30 & close > EMA200
  空: close > BB_upper(20,2σ) & RSI>70 & close < EMA200
  离场: close 回到布林中轨
  止损: -0.30 保证金口径(3x) ≈ 价格 -10%
  仓位: 净值 10%, 杠杆 3x, 最大 4 并发
  费: maker 0.02%

freqtrade 已知结果 (20220101-, 27币, maker 0.02%):
  +105.93% | 2909 trades | Sharpe 1.49 | PF 1.16 | DD 23.82% | avg duration 4:04

用法: python scripts/engine/validate_mr.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_engine import Bar, Portfolio, DataFeed, BacktestEngine, Strategy, generate_report

ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/engine/ → 项目根
D = ROOT / "user_data" / "data" / "binance" / "futures"
import json

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
PAIRS = [p.split("/")[0] + "_USDT_USDT" for p in CONFIG["exchange"]["pair_whitelist"]]


class MRStrategy(Strategy):
    """布林+RSI 均值回归 — 与 freqtrade MeanReversionStrategy 完全相同的规则"""

    def __init__(self, rsi_entry=30, bb_period=20, bb_std=2.0,
                 ema_period=200, stoploss_price=0.10,  # 价格口径 10%
                 stake_pct=0.10, max_open=4):
        self.rsi_entry = rsi_entry
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_period = ema_period
        self.stoploss_price = stoploss_price
        self.stake_pct = stake_pct
        self.max_open = max_open
        # 预计算指标缓存: {pair: DataFrame}
        self._indicators = {}

    def precompute(self, data_dir: Path, pairs: list[str]):
        """预计算所有币的技术指标（严格只用过去数据）"""
        for pair in pairs:
            f = data_dir / f"{pair}-15m-futures.feather"
            if not f.exists():
                continue
            df = pd.read_feather(f).set_index("date")
            c = df["close"]

            mid = c.rolling(self.bb_period).mean()
            sd = c.rolling(self.bb_period).std()
            df["bb_lower"] = mid - self.bb_std * sd
            df["bb_upper"] = mid + self.bb_std * sd
            df["bb_mid"] = mid

            # RSI(14)
            delta = c.diff()
            gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-12))

            df["ema200"] = c.ewm(span=self.ema_period, adjust=False).mean()

            self._indicators[pair] = df

    def on_bar(self, bars: dict[str, Bar], portfolio: Portfolio, ctx: dict):
        date = list(bars.values())[0].date if bars else None
        if date is None:
            return

        for pair, bar in bars.items():
            ind = self._indicators.get(pair)
            if ind is None or date not in ind.index:
                continue

            row = ind.loc[date]
            if pd.isna(row["bb_lower"]) or pd.isna(row["rsi"]) or pd.isna(row["ema200"]):
                continue

            pos = portfolio.positions.get(pair)

            # === 止损检查（用当根 low/high）===
            if pos is not None:
                if pos.side == "long" and bar.low <= pos.entry_price * (1 - self.stoploss_price):
                    portfolio.close(bar, reason="stoploss")
                    continue
                if pos.side == "short" and bar.high >= pos.entry_price * (1 + self.stoploss_price):
                    portfolio.close(bar, reason="stoploss")
                    continue

            # === 离场信号 ===
            if pos is not None:
                if pos.side == "long" and bar.close >= row["bb_mid"]:
                    portfolio.close(bar, reason="mean_reversion")
                    continue
                if pos.side == "short" and bar.close <= row["bb_mid"]:
                    portfolio.close(bar, reason="mean_reversion")
                    continue

            # === 入场信号 ===
            if pos is None and portfolio.n_positions < self.max_open:
                size = portfolio.equity * self.stake_pct
                if bar.close < row["bb_lower"] and row["rsi"] < self.rsi_entry and bar.close > row["ema200"]:
                    portfolio.open_long(bar, size=size)
                elif bar.close > row["bb_upper"] and row["rsi"] > 100 - self.rsi_entry and bar.close < row["ema200"]:
                    portfolio.open_short(bar, size=size)


def main():
    t0 = time.time()

    # === 加载数据 ===
    print("加载数据...")
    feed = DataFeed(D, PAIRS, "15m")
    print(f"  {len(feed._frames)} 币, {feed.total_bars} 根 bar")

    # === 预计算指标 ===
    print("预计算指标...")
    strategy = MRStrategy()
    strategy.precompute(D, PAIRS)
    print(f"  {len(strategy._indicators)} 币有指标")

    # === 运行回测 ===
    print("运行回测...")
    portfolio = Portfolio(
        initial_capital=150,
        fee_maker=0.0002,   # maker 0.02%
        fee_taker=0.0005,
        leverage=3.0,
        funding_interval_bars=32,  # 8h
    )
    engine = BacktestEngine(feed, portfolio, strategy)
    result = engine.run()

    # === 报告 ===
    report = generate_report(result)
    print(f"\n{'='*60}")
    print("引擎回测结果 (MR 规则)")
    print(f"{'='*60}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    # === 对比 freqtrade ===
    print(f"\n{'='*60}")
    print("对比 freqtrade 已知结果")
    print(f"{'='*60}")
    ft = {
        "total_return_pct": 105.93,
        "total_trades": 2909,
        "sharpe": 1.49,
        "max_drawdown_pct": 23.82,
    }
    print(f"{'指标':20s} {'引擎':>12s} {'freqtrade':>12s} {'差异':>10s}")
    for key in ("total_return_pct", "total_trades", "sharpe", "max_drawdown_pct"):
        e = report.get(key, 0)
        f = ft.get(key, 0)
        diff = e - f if isinstance(e, (int, float)) and isinstance(f, (int, float)) else "N/A"
        if isinstance(diff, float):
            print(f"{key:20s} {e:>12.2f} {f:>12.2f} {diff:>+10.2f}")
        else:
            print(f"{key:20s} {e:>12} {f:>12} {'':>10}")

    print(f"\n耗时: {time.time()-t0:.0f}s")

    # 判定
    ret_diff = abs(report.get("total_return_pct", 0) - ft["total_return_pct"])
    trade_ratio = report.get("total_trades", 0) / max(ft["total_trades"], 1)
    if ret_diff < 50 and 0.5 < trade_ratio < 2.0:
        print("\n✅ 引擎与 freqtrade 结果在合理范围内一致 — 引擎可信")
    else:
        print(f"\n⚠️ 差异较大: 收益差{ret_diff:.0f}%, 交易数比{trade_ratio:.2f}")
        print("   可能原因: 指标计算差异 / 离场时机差异 / 仓位管理差异")
        print("   这本身是有价值的诊断信息")


if __name__ == "__main__":
    main()

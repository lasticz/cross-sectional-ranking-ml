# -*- coding: utf-8 -*-
"""可靠回测引擎 — 事件驱动，确定性，可验证

设计原则:
  1. 逐 bar 顺序处理，严格无前视
  2. 同一输入永远产出同一结果（无随机性）
  3. 每个环节可通过测试验证（已知答案）
  4. 回测与实盘用同一套策略接口

架构:
  DataFeed → Strategy.on_bar() → Portfolio.execute() → Reporter

用法:
  from backtest_engine import BacktestEngine, Strategy, Bar

  class MyStrategy(Strategy):
      def on_bar(self, bar, portfolio):
          if bar.close < bar.ema20:
              portfolio.open_long(bar.pair, bar.date, bar.next_open, size=100)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


# ───────────────────────── 数据结构 ───────────────────────── #

@dataclass
class Bar:
    """单根 K 线"""
    pair: str
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    next_open: float = 0.0  # 下一根 bar 的开盘价（由 DataFeed 填充）


@dataclass
class Position:
    """持仓"""
    pair: str
    side: str          # "long" / "short"
    entry_date: pd.Timestamp
    entry_price: float
    size: float        # 名义 USDT
    leverage: float
    unrealized_pnl: float = 0.0
    bars_held: int = 0
    funding_paid: float = 0.0

    @property
    def margin(self) -> float:
        return self.size / self.leverage

    def mark_to_market(self, price: float) -> float:
        """返回未实现盈亏(USDT)"""
        if self.side == "long":
            self.unrealized_pnl = (price / self.entry_price - 1) * self.size
        else:
            # 空头: (1 - price/entry) × size，而非 (entry/price - 1)
            self.unrealized_pnl = (1 - price / self.entry_price) * self.size
        return self.unrealized_pnl


@dataclass
class Trade:
    """已平仓交易"""
    pair: str
    side: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    leverage: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    bars_held: int
    exit_reason: str

    @property
    def ret_pct(self) -> float:
        return self.net_pnl / (self.size / self.leverage)  # 相对保证金


# ───────────────────────── 数据源 ───────────────────────── #

class DataFeed:
    """按 pair 加载 K 线，按时间戳对齐后逐 bar 供给"""

    def __init__(self, data_dir: Path, pairs: list[str], timeframe: str = "15m"):
        self._frames = {}
        self._pairs = pairs
        for p in pairs:
            # p 格式: "ETH_USDT_USDT" → 文件: "ETH_USDT_USDT-15m-futures.feather"
            f = data_dir / f"{p}-{timeframe}-futures.feather"
            if f.exists():
                df = pd.read_feather(f).set_index("date")[["open", "high", "low", "close", "volume"]]
                # 预填 next_open: 下一根 bar 的 open
                df["next_open"] = df["open"].shift(-1)
                self._frames[p] = df

        if not self._frames:
            raise ValueError("未加载到任何数据")

        # 对齐所有 pair 的时间戳
        all_dates = sorted(set().union(*(set(df.index) for df in self._frames.values())))
        self._dates = pd.DatetimeIndex(all_dates)
        self._current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Bar]:
        """返回当前时间戳的所有 pair 的 Bar，{pair: Bar}"""
        if self._current_idx >= len(self._dates):
            raise StopIteration

        date = self._dates[self._current_idx]
        bars = {}
        for pair, df in self._frames.items():
            if date in df.index:
                row = df.loc[date]
                bars[pair] = Bar(
                    pair=pair, date=date,
                    open=row["open"], high=row["high"],
                    low=row["low"], close=row["close"],
                    volume=row["volume"],
                    next_open=row["next_open"] if not np.isnan(row.get("next_open", np.nan)) else row["close"],
                )
        self._current_idx += 1
        return bars

    @property
    def total_bars(self) -> int:
        return len(self._dates)


# ───────────────────────── 组合管理 ───────────────────────── #

class Portfolio:
    """持仓管理与订单执行"""

    def __init__(self, initial_capital: float, fee_maker: float = 0.0002,
                 fee_taker: float = 0.0005, leverage: float = 3.0,
                 funding_interval_bars: int = 32):  # 32×15m = 8h
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.fee_maker = fee_maker
        self.fee_taker = fee_taker
        self.leverage = leverage
        self.funding_interval = funding_interval_bars
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[Trade] = []
        self._bar_count = 0
        self.equity_history: list[dict] = []

    @property
    def equity(self) -> float:
        """当前净值 = 现金 + 所有持仓的未实现盈亏"""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.capital + unrealized

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    def open_long(self, bar: Bar, size: float | None = None):
        """开多: 用下一根开盘价成交"""
        if bar.pair in self.positions or bar.next_open <= 0:
            return
        if size is None:
            size = self.equity / 6  # 默认 1/6 净值
        entry_price = bar.next_open
        fee = size * self.fee_maker
        self.capital -= fee
        self.positions[bar.pair] = Position(
            pair=bar.pair, side="long",
            entry_date=bar.date, entry_price=entry_price,
            size=size, leverage=self.leverage,
        )

    def open_short(self, bar: Bar, size: float | None = None):
        """开空"""
        if bar.pair in self.positions or bar.next_open <= 0:
            return
        if size is None:
            size = self.equity / 6
        entry_price = bar.next_open
        fee = size * self.fee_maker
        self.capital -= fee
        self.positions[bar.pair] = Position(
            pair=bar.pair, side="short",
            entry_date=bar.date, entry_price=entry_price,
            size=size, leverage=self.leverage,
        )

    def close(self, bar: Bar, reason: str = "signal"):
        """平仓: 用下一根开盘价"""
        if bar.pair not in self.positions:
            return
        pos = self.positions.pop(bar.pair)
        exit_price = bar.next_open if bar.next_open > 0 else bar.close

        if pos.side == "long":
            gross = (exit_price / pos.entry_price - 1) * pos.size
        else:
            gross = (1 - exit_price / pos.entry_price) * pos.size

        fee = pos.size * self.fee_maker  # 平仓费
        net = gross - fee - pos.funding_paid
        self.capital += net

        self.closed_trades.append(Trade(
            pair=pos.pair, side=pos.side,
            entry_date=pos.entry_date, exit_date=bar.date,
            entry_price=pos.entry_price, exit_price=exit_price,
            size=pos.size, leverage=pos.leverage,
            gross_pnl=gross, fees=fee + pos.size * self.fee_maker,
            funding=pos.funding_paid, net_pnl=net,
            bars_held=pos.bars_held, exit_reason=reason,
        ))

    def close_all(self, bars: dict[str, Bar], reason: str = "end"):
        for pair in list(self.positions.keys()):
            if pair in bars:
                self.close(bars[pair], reason)

    def on_bar_end(self, bars: dict[str, Bar]):
        """每根 bar 结束时: mark-to-market + 资金费 + 记录净值"""
        self._bar_count += 1

        # 资金费: 每 funding_interval 根 bar 结算一次
        if self._bar_count % self.funding_interval == 0:
            for pos in self.positions.values():
                if pos.pair in bars:
                    # 资金费简化: 假设费率为 0.01%/8h，多头付费
                    funding_rate = 0.0001
                    if pos.side == "long":
                        pos.funding_paid += pos.size * funding_rate
                    else:
                        pos.funding_paid -= pos.size * funding_rate

        # Mark to market
        for pair, pos in self.positions.items():
            if pair in bars:
                pos.mark_to_market(bars[pair].close)
                pos.bars_held += 1

        # 记录净值
        self.equity_history.append({
            "bar": self._bar_count,
            "equity": self.equity,
            "n_positions": self.n_positions,
        })


# ───────────────────────── 策略基类 ───────────────────────── #

class Strategy:
    """策略基类: 继承并实现 on_bar"""

    def on_bar(self, bar: Bar, portfolio: Portfolio, context: dict):
        """每根 bar 调用一次。通过 portfolio.open_long/close 等方法交易。"""
        raise NotImplementedError

    def on_start(self, portfolio: Portfolio):
        """回测开始时调用一次"""
        pass

    def on_end(self, portfolio: Portfolio):
        """回测结束时调用一次"""
        pass


# ───────────────────────── 引擎 ───────────────────────── #

class BacktestEngine:
    """事件驱动回测引擎"""

    def __init__(self, data_feed: DataFeed, portfolio: Portfolio, strategy: Strategy):
        self.feed = data_feed
        self.portfolio = portfolio
        self.strategy = strategy

    def run(self) -> Portfolio:
        self.strategy.on_start(self.portfolio)

        # 预计算指标（策略自己管理）
        context = {}

        for bars in self.feed:
            # 1. 策略决策
            self.strategy.on_bar(bars, self.portfolio, context)

            # 2. 逐 bar 结算
            self.portfolio.on_bar_end(bars)

        # 平掉所有剩余仓位
        last_bars = {}
        # 重新迭代最后一天
        self.strategy.on_end(self.portfolio)
        self.portfolio.close_all(last_bars, "end_of_backtest")

        return self.portfolio


# ───────────────────────── 报告 ───────────────────────── #

def generate_report(portfolio: Portfolio) -> dict:
    """从 Portfolio 生成绩效报告"""
    trades = portfolio.closed_trades
    equity = pd.DataFrame(portfolio.equity_history)

    if not trades:
        return {"total_trades": 0}

    rets = [t.ret_pct for t in trades]
    eq = equity["equity"].to_numpy()

    # 最大回撤
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min() if len(dd) > 0 else 0

    # Sharpe (日收益近似)
    if len(eq) > 100:
        daily_ret = np.diff(eq) / eq[:-1]
        # 重采样到日
        sharpe = np.mean(daily_ret) / (np.std(daily_ret) + 1e-12) * np.sqrt(365 * 96)
    else:
        sharpe = 0

    total_pnl = sum(t.net_pnl for t in trades)
    total_fees = sum(t.fees for t in trades)

    return {
        "total_trades": len(trades),
        "win_rate": sum(1 for t in trades if t.net_pnl > 0) / len(trades),
        "avg_ret_bps": np.mean(rets) * 10000,
        "median_ret_bps": np.median(rets) * 10000,
        "total_pnl_usdt": total_pnl,
        "total_fees_usdt": total_fees,
        "final_equity": portfolio.equity,
        "total_return_pct": (portfolio.equity / portfolio.initial_capital - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "avg_hold_hours": np.mean([t.bars_held for t in trades]) * 15 / 60,
    }

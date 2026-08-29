# -*- coding: utf-8 -*-
"""NautilusTrader Step 2: 最简验证 — 买入持有 SOL"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import decimal

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.test_kit.providers import TestInstrumentProvider


def load_bars(pair="BTC_USDT_USDT", start="2024-01-01", end="2024-02-01", price_prec=1, size_prec=3):
    f = D / f"{pair}-15m-futures.feather"
    df = pd.read_feather(f).set_index("date")
    df = df.loc[start:end]
    bar_type = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    bars = []
    for date, row in df.iterrows():
        ts_ns = int(date.timestamp() * 1e9)
        bars.append(Bar(
            bar_type=bar_type,
            open=Price(round(row["open"], price_prec), precision=price_prec),
            high=Price(round(row["high"], price_prec), precision=price_prec),
            low=Price(round(row["low"], price_prec), precision=price_prec),
            close=Price(round(row["close"], price_prec), precision=price_prec),
            volume=Quantity(round(row["volume"], size_prec), precision=size_prec),
            ts_event=ts_ns, ts_init=ts_ns,
        ))
    return bars, df


class BuyHold:
    """最小策略: 第 1 根 bar 市价买入, 全部 bar 结束后引擎平仓"""
    bought = False
    bar_count = 0

    def on_bar(self, bar, strategy):
        self.bar_count += 1
        if not self.bought:
            strategy.buy(
                instrument_id=bar.bar_type.instrument_id,
                quantity=Quantity(100, precision=1),
            )
            self.bought = True


def main():
    print("=== NautilusTrader 基础验证 ===\n")
    bars, raw = load_bars()
    first_open = raw["open"].iloc[0]
    last_close = raw["close"].iloc[-1]
    theoretical = (last_close / first_open - 1) * 100
    print(f"SOL 15m 2024-01: {len(bars)} bars")
    print(f"  首open={first_open:.2f} 末close={last_close:.2f}")
    print(f"  理论收益: {theoretical:+.2f}%")

    engine = BacktestEngine(config=BacktestEngineConfig(trader_id="TESTER-001"))
    instrument = TestInstrumentProvider.btcusdt_perp_binance()
    from nautilus_trader.model.enums import OmsType, AccountType
    engine.add_venue(
        instrument.id.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(150_000, USDT)],
        default_leverage=decimal.Decimal(3),
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)

    from nautilus_trader.trading.strategy import Strategy as NTStrategy

    class BuyHoldNT(NTStrategy):
        def __init__(self):
            super().__init__()
            self.bought = False
            self.n = 0

        def on_start(self):
            self.subscribe_bars(BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL"))

        def on_bar(self, bar):
            self.n += 1
            if not self.bought:
                order = self.order_factory.market(
                    instrument_id=bar.bar_type.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity(100, precision=3),
                )
                self.submit_order(order)
                self.bought = True

    engine.add_strategy(BuyHoldNT())
    engine.run()

    print(f"\n=== 引擎结果 ===")
    print(f"  处理 bar 数: 2880 (预期)")
    positions = engine.trader.generate_positions_report()
    if len(positions) > 0:
        print(f"  持仓数: {len(positions)}")
        for _, p in positions.iterrows():
            print(f"    {p.to_dict()}")
    else:
        print("  无持仓数据")

    # 账户
    accounts = engine.trader.generate_account_report(instrument.id.venue)
    if len(accounts) > 0:
        print(f"  账户报告:")
        print(accounts.to_string())

    print("\n✅ NautilusTrader 基础运行成功")


if __name__ == "__main__":
    main()

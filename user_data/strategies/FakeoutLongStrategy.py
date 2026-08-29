# -*- coding: utf-8 -*-
"""深度假突破做多策略（15m，只做多，持仓 1-2h）

依据: 事件研究（RESEARCH.md 2026-08-27）——下影刺穿前 20 根低点 >=0.5% 且收盘收回该位置上方 >=0.5%
     （深度扫荡+强力收回），下一根开盘做多持 1h:
     train +8.4bps (胜率55%, t=2.9) / test +16.6bps (t=3.9)，费后现实入场口径。
     空头方向（扫高点做空）越深越亏，已否决。
与 FlashCrashStrategy 互补: 那个只看跌幅幅度，这个要求当根内收回确认（买方防守成功）。
"""
import os
from datetime import datetime, timedelta

from pandas import DataFrame

from freqtrade.strategy import IStrategy


class FakeoutLongStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short: bool = False

    minimal_roi = {"0": 100}
    # freqtrade 合约止损为保证金口径: -0.15 在 3x 下 ≈ 价格 -5%，纯灾难兜底
    stoploss = -0.15

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # 刺穿/收回深度（相对 20 根低点），支持环境变量注入供 walk-forward
    depth = float(os.environ.get("FO_DEPTH", "0.005"))
    hold = timedelta(hours=float(os.environ.get("FO_HOLD", "1")))

    startup_candle_count: int = 60

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side, **kwargs) -> float:
        return 3.0

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag, side, **kwargs) -> float:
        pct = float(os.environ.get("STAKE_PCT", "0.10"))
        stake = self.wallets.get_total_stake_amount() * pct
        if min_stake:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        if current_time - trade.open_date_utc >= self.hold:
            return "hold_timeout"
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["pl20"] = dataframe["low"].rolling(20).min().shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["low"] < dataframe["pl20"] * (1 - self.depth))
            & (dataframe["close"] > dataframe["pl20"] * (1 + self.depth))
            & (dataframe["volume"] > 0),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 注意: exit 条件不可与 entry 同根并存（freqtrade 会丢弃 entry）——本策略离场全靠超时
        return dataframe

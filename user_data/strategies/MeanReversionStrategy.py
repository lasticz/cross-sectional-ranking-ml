# -*- coding: utf-8 -*-
"""布林带均值回归策略（15m，合约双向）

15m 周期的适配策略类型: 超跌/超涨的短期回归。
入场:
  多: 收盘跌破布林下轨(20,2σ) 且 RSI<25 且 价格在 EMA200 上方（不在瀑布里接刀）
  空: 收盘突破布林上轨 且 RSI>75 且 价格在 EMA200 下方
离场: 收盘回到布林中轨（回归完成）
仓位: 净值 3% 保证金 × 3x；止损 -0.30 保证金口径（≈价格 10%）纯兜底。
"""
import os
from datetime import datetime
from pandas import DataFrame

import talib.abstract as ta
from technical import qtpylib

from freqtrade.strategy import IStrategy


class MeanReversionStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short: bool = True

    minimal_roi = {"0": 100}
    stoploss = -0.30

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 2400

    # 入场 RSI 阈值（多头 < rsi，空头 > 100-rsi），支持环境变量注入供 walk-forward
    rsi_entry = int(os.environ.get("MR_RSI", "30"))

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
        # 保证金占净值比例，支持环境变量注入；激进档默认 10%
        pct = float(os.environ.get("STAKE_PCT", "0.10"))
        stake = self.wallets.get_total_stake_amount() * pct
        if min_stake:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi"] < self.rsi_entry)
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["volume"] > 0),
            "enter_long"] = 1
        dataframe.loc[
            (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["rsi"] > 100 - self.rsi_entry)
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["volume"] > 0),
            "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] >= dataframe["bb_mid"]) & (dataframe["volume"] > 0),
            "exit_long"] = 1
        dataframe.loc[
            (dataframe["close"] <= dataframe["bb_mid"]) & (dataframe["volume"] > 0),
            "exit_short"] = 1
        return dataframe

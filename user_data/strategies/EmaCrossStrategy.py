# --- Strategy libs ---
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)
from technical import qtpylib


class EmaCrossStrategy(IStrategy):
    """
    基础双 EMA 交叉趋势策略（1h）

    入场: 快线 EMA 上穿慢线 EMA，且 RSI > 50（动量确认）、ADX > 阈值（趋势强度过滤）
    出场: 快线 EMA 下穿慢线 EMA
    风控: 固定止损 -10%，移动止损（+5% 后启动，回撤 2.5% 触发）
    """
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short: bool = False

    # 到达 +10% 直接止盈落袋
    minimal_roi = {
        "0": 0.10
    }

    stoploss = -0.10

    # 移动止损: 浮盈超过 5% 后启动，从高点回撤 2.5% 离场
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 120

    # 可 hyperopt 调优的参数
    # hyperopt (train 2023-2024, SharpeHyperOptLoss) 最优值
    ema_fast = IntParameter(10, 30, default=24, space="buy")
    ema_slow = IntParameter(40, 120, default=76, space="buy")
    rsi_threshold = IntParameter(45, 60, default=45, space="buy")
    adx_threshold = IntParameter(15, 35, default=34, space="buy")

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["ema_fast"], dataframe["ema_slow"])
                & (dataframe["rsi"] > self.rsi_threshold.value)
                & (dataframe["adx"] > self.adx_threshold.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe["ema_fast"], dataframe["ema_slow"])
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe

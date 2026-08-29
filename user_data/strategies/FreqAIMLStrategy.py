# -*- coding: utf-8 -*-
"""FreqAI 机器学习策略（15m，LightGBM 回归，预测 1h 前瞻收益）

特征: 30+ 个（价格动量/布林位置/RSI/成交量/波动率/横截面/时段/资金费）
目标: 未来 4 根 15m（1h）收益率（回归）
入场: 预测收益 > 0.15% 做多；预测收益 < -0.15% 做空
离场: 持有 1h 超时
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class FreqAIMLStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short: bool = True

    minimal_roi = {"0": 100}
    stoploss = -0.15  # 保证金口径(3x) ≈ 价格 -5%，兜底

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 400

    # FreqAI 会覆盖这些参数
    # 以下定义特征工程，FreqAI 自动收集并展开到所有币
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                        metadata: dict, **kwargs) -> DataFrame:
        """每个币独立计算的特征（period 参数自动展开多周期）"""
        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-roc"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-atr"] = ta.ATR(dataframe, timeperiod=period) / dataframe["close"]
        dataframe["%-bb-pos"] = (dataframe["close"] - dataframe["close"].rolling(period).mean()) / (
            2 * dataframe["close"].rolling(period).std() + 1e-12
        )
        dataframe["%-vol-ratio"] = dataframe["volume"] / (
            dataframe["volume"].rolling(period * 4).mean() + 1e-12
        )
        dataframe["%-dist-ema"] = dataframe["close"] / dataframe["close"].ewm(span=period * 10).mean() - 1
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, period: int,
                                          metadata: dict, **kwargs) -> DataFrame:
        """跨币特征（FreqAI 自动计算所有币的均值）"""
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """标准特征（不按 period 展开，每个币独立）"""
        dataframe["%-hour"] = dataframe["date"].dt.hour
        dataframe["%-dow"] = dataframe["date"].dt.dayofweek
        dataframe["%-bb-width"] = (
            (4 * dataframe["close"].rolling(20).std()) /
            (dataframe["close"].rolling(20).mean() + 1e-12)
        )
        dataframe["%-body-ratio"] = np.abs(dataframe["close"] - dataframe["open"]) / (
            dataframe["high"] - dataframe["low"] + 1e-12
        )
        # 高时间框架趋势
        dataframe["%-htf-trend"] = (
            dataframe["close"].ewm(span=200).mean() >
            dataframe["close"].ewm(span=800).mean()
        ).astype(int)
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """目标: 未来 4 根 15m（1h）收益率"""
        dataframe["&-ret-1h"] = (
            dataframe["close"].shift(-4) / dataframe["close"] - 1
        )
        return dataframe

    # ===== 交易逻辑 =====

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # FreqAI 会自动调用 feature_engineering 并附加预测列
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # FreqAI 注入预测列到 dataframe；列名 = set_freqai_targets 中定义的目标名
        if "&-ret-1h" not in dataframe.columns:
            return dataframe
        pred = dataframe["&-ret-1h"]
        # 调试: 先用极低阈值看是否有信号
        dataframe.loc[
            (pred > 0.0001) & (dataframe["volume"] > 0),
            "enter_long"] = 1
        dataframe.loc[
            (pred < -0.0001) & (dataframe["volume"] > 0),
            "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 持有 1h 超时由 custom_exit 处理，这里不加信号离场
        return dataframe

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        if current_time - trade.open_date_utc >= timedelta(hours=1):
            return "hold_timeout"
        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return 3.0

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        return min(proposed_stake, self.wallets.get_total_stake_amount() / 6)

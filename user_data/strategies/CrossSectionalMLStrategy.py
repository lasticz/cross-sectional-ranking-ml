# -*- coding: utf-8 -*-
"""横截面 ML Top3/Bottom3 策略（freqtrade 信号桥模式）

信号来源: user_data/ml_signals.json（由 04_freqtrade_bridge.py 生成）
每个 4h 窗口: 做多预测 Top3 + 做空预测 Bottom3
离场: 信号窗口结束（下一个 4h 窗口开始时，新排名自动换仓）
杠杆: 3x, 保证金: 净值/6
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class CrossSectionalMLStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short: bool = True

    minimal_roi = {"0": 100}
    stoploss = -0.30  # 保证金口径(3x) ≈ 价格 -10%, 兜底

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 10

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    max_open_trades = 6  # 3 long + 3 short

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return 3.0

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        return min(proposed_stake, 25.0)

    def _signals(self):
        f = Path = __import__("pathlib").Path
        sig_file = f(__import__("sys").argv[0]).resolve().parent.parent / "user_data" / "ml_signals.json"
        # 更可靠: 用项目根目录
        import pathlib
        sig_file = pathlib.Path.cwd() / "user_data" / "ml_signals.json"
        if not sig_file.exists():
            return []
        try:
            return json.loads(sig_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sigs = [s for s in self._signals() if s.get("pair") == metadata["pair"]]
        dataframe["in_long_window"] = False
        dataframe["in_short_window"] = False
        if not sigs:
            return dataframe

        s = pd.DataFrame(sigs)
        s["ws"] = pd.to_datetime(s["window_start"], utc=True)
        s["we"] = pd.to_datetime(s["window_end"], utc=True)
        s = s.sort_values("ws").reset_index(drop=True)

        # 统一转 datetime64[ns, UTC] 再取秒, 避免ms/us/ns单位混乱
        candle_t = (dataframe["date"].astype("datetime64[ns, UTC]").astype("int64") // 10**9).to_numpy()
        ws_arr = (s["ws"].astype("datetime64[ns, UTC]").astype("int64") // 10**9).to_numpy()
        we_arr = (s["we"].astype("datetime64[ns, UTC]").astype("int64") // 10**9).to_numpy()

        pos = np.searchsorted(ws_arr, candle_t, side="right") - 1
        valid = pos >= 0
        v_idx = np.where(valid)[0]

        is_long = np.zeros(len(dataframe), dtype=bool)
        is_short = np.zeros(len(dataframe), dtype=bool)

        if len(v_idx) > 0:
            side_arr = s["side"].to_numpy()
            for i in v_idx:
                p = pos[i]
                if candle_t[i] < we_arr[p]:
                    if side_arr[p] == "long":
                        is_long[i] = True
                    else:
                        is_short[i] = True

        dataframe["in_long_window"] = is_long
        dataframe["in_short_window"] = is_short
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            dataframe["in_long_window"] & (dataframe["volume"] > 0),
            "enter_long"] = 1
        dataframe.loc[
            dataframe["in_short_window"] & (dataframe["volume"] > 0),
            "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 信号窗口结束时离场（新窗口的新币自动通过 enter 信号开仓）
        dataframe.loc[
            ~dataframe["in_long_window"] & (dataframe["volume"] > 0),
            "exit_long"] = 1
        dataframe.loc[
            ~dataframe["in_short_window"] & (dataframe["volume"] > 0),
            "exit_short"] = 1
        return dataframe

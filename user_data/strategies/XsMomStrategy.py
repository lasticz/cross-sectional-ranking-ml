# -*- coding: utf-8 -*-
"""24h 横截面动量策略（1h，合约双向，市场中性）

依据: 事件研究（RESEARCH.md 2026-08-27）——27 币按过去 24h 收益排名，
     多最强3+空最弱3，日换仓: train +66%/yr, test +64%/yr（费后粗算，胜率48%尾部驱动）。
机制: 每天 00:00 UTC 评估，排名入前K做多/入后K做空；跌出对应区间才离场（日内不动作）。
仓位: 每条腿名义 = 净值/6（3x 杠杆下保证金 ≈5.5%），双腿合计名义 ≈100% 净值。
"""
import os
from datetime import datetime

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter


class XsMomStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short: bool = True

    minimal_roi = {"0": 100}
    # 保证金口径 -0.90 ≈ 价格 -30%（3x），贴强平线的兜底；研究口径无止损
    stoploss = -0.90

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # 回看窗口（小时）与持仓数 K，支持环境变量注入供 walk-forward
    lookback_h = int(os.environ.get("XSM_LB", "24"))
    k = int(os.environ.get("XSM_K", "3"))
    eval_hour = 0  # UTC 评估时刻

    startup_candle_count: int = 200

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    max_open_trades = 6

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side, **kwargs) -> float:
        return 3.0

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag, side, **kwargs) -> float:
        # 名义 = 净值/6；3x 下保证金 ≈ 净值/18
        stake = self.wallets.get_total_stake_amount() / (6 * 3)
        if min_stake:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lb_bars = self.lookback_h  # 1h 周期
        rets = {}
        for p in self.config["exchange"]["pair_whitelist"]:
            inf = self.dp.get_pair_dataframe(p, self.timeframe)
            if inf is None or inf.empty:
                continue
            s = inf.set_index("date")["close"]
            rets[p] = s / s.shift(lb_bars) - 1.0
        wide = pd.DataFrame(rets)
        ranks = wide.rank(axis=1, ascending=False, na_option="keep")  # 1=最强
        n = wide.notna().sum(axis=1)
        dataframe = dataframe.merge(
            ranks[[metadata["pair"]]].rename(columns={metadata["pair"]: "rank"})
            .join(n.rename("n")).reset_index(),
            on="date", how="left",
        )
        # 每日重建: 0点算排名(shift 1 根让信号落在1点bar), 次日0点无条件全平
        dataframe["rank"] = dataframe["rank"].shift(1)
        dataframe["n"] = dataframe["n"].shift(1)
        dataframe["eval_bar"] = dataframe["date"].dt.hour == (self.eval_hour + 1)
        dataframe["close_bar"] = dataframe["date"].dt.hour == self.eval_hour
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ok = dataframe["eval_bar"] & (dataframe["n"] >= 10) & (dataframe["volume"] > 0)
        dataframe.loc[ok & (dataframe["rank"] <= self.k), "enter_long"] = 1
        dataframe.loc[ok & (dataframe["rank"] >= dataframe["n"] - self.k + 1), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 每日重建: 0点bar无条件全平（1点重进），与入场bar错开无冲突
        dataframe.loc[dataframe["close_bar"] & (dataframe["volume"] > 0), ["exit_long", "exit_short"]] = 1
        return dataframe

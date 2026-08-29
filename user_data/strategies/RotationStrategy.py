# -*- coding: utf-8 -*-
"""横截面动量轮动（合约双向，市场中性）

逻辑: 每 4h 对白名单 10 个币按过去 N 根 bar 的收益率排名，
      做多最强 2 个、做空最弱 2 个；排名离开缓冲区才换仓（防抖）。
仓位: 每笔 = 当前净值的 3% 保证金，杠杆固定 3x（见 custom_stake_amount / leverage）。
风控: 灾难止损 -25%（价格口径，3x 下约亏 75% 保证金）；常规离场靠排名变化。
"""
import os

import pandas as pd
from pandas import DataFrame
from datetime import datetime

from freqtrade.strategy import IStrategy, IntParameter, timeframe_to_minutes


class RotationStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    can_short: bool = True

    # ROI 实际关闭；止损为保证金口径 -60% ≈ 价格 -20%（3x），纯灾难兜底
    minimal_roi = {"0": 100}
    stoploss = -0.60

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 动量回看窗口（自然日），周期无关；regime EMA 固定 30 天
    # 默认值支持环境变量注入，供 walk-forward 脚本逐窗选参
    lookback_days = IntParameter(1, 14, default=int(os.environ.get("ROT_LOOKBACK_DAYS", "7")), space="buy")
    # 触发信号所需的最少可用币数（SUI 2023-05 才上线，早期横截面窄）
    min_universe = 8
    regime_days = 30

    @property
    def startup_candle_count(self) -> int:
        bars_per_day = 60 * 24 // timeframe_to_minutes(self.timeframe)
        # 2400 为 freqtrade 对 15m 的校验上限；超出部分由指标 NaN 自然延迟信号
        return min((14 + self.regime_days + 1) * bars_per_day, 2400)

    @property
    def bars_per_day(self) -> int:
        return 60 * 24 // timeframe_to_minutes(self.timeframe)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def informative_pairs(self):
        return [(p, self.timeframe) for p in self.config["exchange"]["pair_whitelist"]]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side, **kwargs) -> float:
        return 3.0

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag, side, **kwargs) -> float:
        # 浮动 3%: 按当前总净值计算保证金
        stake = self.wallets.get_total_stake_amount() * 0.03
        if min_stake:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        roc_by_pair = {}
        norm_close = {}
        for p in self.config["exchange"]["pair_whitelist"]:
            inf = self.dp.get_pair_dataframe(p, self.timeframe)
            if inf is None or inf.empty:
                continue
            s = inf.set_index("date")["close"]
            roc_by_pair[p] = s / s.shift(self.lookback_days.value * self.bars_per_day) - 1.0
            # 归一化价格用于构造宇宙等权指数（每币从 1 起步）
            norm_close[p] = s / s.iloc[0]

        wide = pd.DataFrame(roc_by_pair)
        # rank 1 = 动量最强；当根 bar 缺数据的币不参与排名
        ranks = wide.rank(axis=1, ascending=False, na_option="keep")
        n_avail = wide.notna().sum(axis=1)

        # 市场状态: 等权指数 vs 其 30 天 EMA
        span = self.regime_days * self.bars_per_day
        index = pd.DataFrame(norm_close).mean(axis=1)
        regime_up = (index > index.ewm(span=span, min_periods=span).mean()).astype(float)

        dataframe = dataframe.merge(
            ranks[[metadata["pair"]]].rename(columns={metadata["pair"]: "rank"})
            .join(n_avail.rename("n_avail")).join(regime_up.rename("regime_up")).reset_index(),
            on="date", how="left",
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        n = dataframe["n_avail"]
        ok = (n >= self.min_universe) & (dataframe["volume"] > 0)
        # 市场状态门控: 指数在长期均线上方只做多，下方只做空
        dataframe.loc[ok & (dataframe["rank"] <= 2) & (dataframe["regime_up"] > 0), "enter_long"] = 1
        dataframe.loc[ok & (dataframe["rank"] >= n - 1) & (dataframe["regime_up"] == 0), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        n = dataframe["n_avail"]
        # 缓冲区: 多头跌到后一半才离场（排名 3-5 持有）；空头爬回前一半才离场
        dataframe.loc[(dataframe["rank"] > 5) & (dataframe["volume"] > 0), "exit_long"] = 1
        dataframe.loc[(dataframe["rank"] < n - 4) & (dataframe["volume"] > 0), "exit_short"] = 1
        return dataframe


class RotationFast(RotationStrategy):
    """1 天动量变体，用于探索更快信号的轮动。"""
    lookback_days = IntParameter(1, 14, default=1, space="buy")

# -*- coding: utf-8 -*-
"""资金费率极端反向策略 v2（1h，合约，只做多）

依据 train 段（2022-2024, SOL/BNB/XRP）事件研究:
  负费率 ≤ -0.05%/8h 事件后 7 天前瞻收益: SOL 中位数 +41.6%, BNB +13.8%
  均值 > 中位数（利润集中在少数大轧空），因此必须拿满持有期
设计:
  入场: 费率 ≤ -0.05%/8h（空头拥挤，做多方同时每 8h 收资金费）
  离场: 持有 168h（7 天，事件研究的最优持有期）或费率 > +0.05%（轧空过热）
仓位: 净值 3% 保证金 × 3x。
注意: 回测中资金费收支由 freqtrade 自动计入 PnL。
"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter


class FundingExtremeStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short: bool = False

    minimal_roi = {"0": 100}   # 关闭 ROI
    stoploss = -0.30           # 保证金口径 ≈ 价格 -10%（3x），灾难兜底

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # 默认值支持环境变量注入，供 walk-forward 脚本逐窗选参
    long_funding = DecimalParameter(-0.002, -0.0001,
                                    default=float(os.environ.get("FUND_TH", "-0.0005")),
                                    decimals=4, space="buy")
    overheat_funding = 0.0005  # 轧空过热止盈线
    max_hold = timedelta(hours=168)

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

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        # 拿满事件研究的持有期；事件内的资金费收入随持仓自动入账
        if current_time - trade.open_date_utc >= self.max_hold:
            return "hold_timeout"
        return None

    def _funding_series(self, pair: str, index: pd.DatetimeIndex) -> pd.Series | None:
        """读取 freqtrade 下载的资金费率文件，前向填充到 1h 网格并滞后一根 bar。

        仅回测可用（读本地文件）；实盘/dry-run 需改走交易所 API（TODO）。
        """
        fname = re.sub(r"[/:]", "_", pair) + "-1h-funding_rate.feather"
        path = Path(self.config["user_data_dir"]) / "data" / "binance" / "futures" / fname
        if not path.exists():
            return None
        df = pd.read_feather(path)[["date", "open"]].rename(columns={"open": "funding"})
        s = df.set_index("date")["funding"].reindex(index.union(df["date"])).ffill()
        return s.reindex(index).shift(1)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        funding = self._funding_series(metadata["pair"], pd.DatetimeIndex(dataframe["date"]))
        dataframe["funding"] = funding.to_numpy() if funding is not None else float("nan")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["funding"] <= self.long_funding.value)
            & (dataframe["volume"] > 0),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 费率转正且过热 = 轧空行情走到拥挤的对面，落袋
        dataframe.loc[
            (dataframe["funding"] > self.overheat_funding)
            & (dataframe["volume"] > 0),
            "exit_long"] = 1
        return dataframe

# -*- coding: utf-8 -*-
"""新币上线阴跌做空策略（1h，合约，只做空）

依据: RESEARCH.md 新币研究——2023-26 共 412 个新上永续，上线后 10 天中位跌幅 6-22%，
     D+1 进空持 10 天: train 中位 +8.3%/胜率63%，test +16.3%/胜率69%（四年全正）。
逻辑:
  入场: 币龄进入 [24h, 72h] 窗口（上线首日情绪定型后）
  离场: 持有 10 天超时，或价格反向 +50% 止损（防妖币）
  杠杆: 固定 1x（尾部风险真实，edge 不需要杠杆放大）
仓位: 净值 10% 保证金（env STAKE_PCT 可调），最多 6 并发。
币龄: 读 user_data/listings.json 的 onboard 日期（live 需改为交易所 markets 的 onboardDate，TODO）。
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from pandas import DataFrame

from freqtrade.strategy import IStrategy


class NewListingShortStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short: bool = True

    minimal_roi = {"0": 100}
    # 1x 下止损即价格反向幅度; 默认 -50%，实验可注入（LIST_SL=-0.30）
    stoploss = float(os.environ.get("LIST_SL", "-0.50"))
    # 只空破发币: 入场时价格低于上市开盘价（LIST_WEAK=1 开启），跳过强势妖币候选
    weak_only = os.environ.get("LIST_WEAK", "0") == "1"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 10
    # 并发槽位可注入（上新扎堆时决定信号捕获率）
    max_open_trades = int(os.environ.get("LIST_SLOTS", "6"))

    entry_age_min = 24   # 小时
    entry_age_max = 72
    max_hold = timedelta(days=10)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._onboard: dict[str, datetime] = {}
        lj = Path(config["user_data_dir"]) / "listings.json"
        if lj.exists():
            for row in json.loads(lj.read_text(encoding="utf-8")):
                # "ABC/USDT:USDT" -> key "ABC"
                self._onboard[row["base"]] = datetime.fromisoformat(row["onboard"]).replace(tzinfo=None)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side, **kwargs) -> float:
        return 1.0

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag, side, **kwargs) -> float:
        pct = float(os.environ.get("STAKE_PCT", "0.10" if int(os.environ.get("LIST_SLOTS", "6")) <= 6 else "0.05"))
        stake = self.wallets.get_total_stake_amount() * pct
        if min_stake:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        if current_time - trade.open_date_utc >= self.max_hold:
            return "hold_timeout"
        return None

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        base = metadata["pair"].split("/")[0]
        ob = self._onboard.get(base)
        if ob is None:
            dataframe["age_h"] = 1e9  # 不在名单里: 永不入场
        else:
            age = (dataframe["date"].dt.tz_localize(None) - ob) / timedelta(hours=1)
            dataframe["age_h"] = age
        # 上市开盘价（对本策略实际会入场的新币，数据首根即上市首根）
        dataframe["d0_open"] = dataframe["open"].iloc[0]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cond = (
            (dataframe["age_h"] >= self.entry_age_min)
            & (dataframe["age_h"] <= self.entry_age_max)
            & (dataframe["volume"] > 0)
        )
        if self.weak_only:
            cond &= dataframe["close"] < dataframe["d0_open"]
        dataframe.loc[cond, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

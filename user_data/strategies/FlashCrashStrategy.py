# -*- coding: utf-8 -*-
"""山寨闪崩反弹策略（15m，合约，只做多，持仓 1-4h）

依据: 事件研究（RESEARCH.md 2026-08-27）——单根 15m 跌超 3% 且 BTC 同根未跌（币自身闪崩）时，
     买进持 1-2h: train +64~94bps/笔 (t=6-7), test +59~70bps (t=3-4), 胜率 60%。
     注意 -5% 以上深崩会继续跌（test 转负），所以阈值取 -3% "够疼不致命"档。
与 BtcShockStrategy 互补: 那个吃市场级冲击(BTC急跌)，这个吃个股级插针(BTC稳定)。
"""
import os
from datetime import datetime, timedelta

from pandas import DataFrame

from freqtrade.strategy import IStrategy

BTC_PAIR = "BTC/USDT:USDT"


class FlashCrashStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short: bool = False

    minimal_roi = {"0": 100}
    stoploss = -0.10  # 价格口径灾难兜底，常规离场靠超时

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # 触发阈值（单根跌幅）；环境变量注入经 crash_th_val 属性（避免类属性名冲突）
    crash_th_val = float(os.environ.get("FC_TH", "-0.03"))
    btc_exclude = -0.01      # BTC 同根跌超 1% 视为市场级冲击，交给 BtcShockStrategy
    hold = timedelta(hours=float(os.environ.get("FC_HOLD", "2")))

    startup_candle_count: int = 50

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def informative_pairs(self):
        return [(BTC_PAIR, self.timeframe)]

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
        dataframe["ret1"] = dataframe["close"].pct_change()
        btc = self.dp.get_pair_dataframe(BTC_PAIR, self.timeframe)
        btc_ret = btc.set_index("date")["close"].pct_change().rename("btc_ret")
        dataframe = dataframe.merge(btc_ret.reset_index(), on="date", how="left")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ret1"] <= self.crash_th_val)
            & (dataframe["btc_ret"] > self.btc_exclude)
            & (dataframe["volume"] > 0),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 二次崩盘不接: 持仓期间再跌超阈值提前离场。
        # 注意必须排除入场信号根（freqtrade 中同根 exit 信号会吞掉 entry）
        crash = (dataframe["ret1"] <= self.crash_th_val) & (dataframe["volume"] > 0)
        entry_bar = (
            (dataframe["ret1"] <= self.crash_th_val)
            & (dataframe["btc_ret"] > self.btc_exclude)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[crash & ~entry_bar, "exit_long"] = 1
        return dataframe

# -*- coding: utf-8 -*-
"""BTC 急跌冲击反弹策略（15m，合约，只做多）

来源: 异常扫描（RESEARCH.md 2026-08-27）。BTC 单根 15m 跌超阈值后，
     山寨在随后 1-2 小时平均反弹（train +13~22bps / test +7.7~7.9bps，maker 费前）。
逻辑:
  触发: BTC 同周期 K 线收益率 < -shock_th（默认 -0.5%）
  选币: 冲击当根跌幅最深的前 3 名（rank <= shock_top_n）
  入场: 信号根收盘后下一根开盘（freqtrade 标准，无前视）
  离场: 持有 2 小时（8 根 15m）超时；止损 -5% 价格口径兜底接刀风险
执行要点: 必须挂单进场（limit），taker 往返 10bps 会吃掉全部 edge。
"""
import os
from datetime import datetime, timedelta
from functools import reduce

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

BTC_PAIR = "BTC/USDT:USDT"
HOLD = timedelta(hours=2)


class BtcShockStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short: bool = False

    minimal_roi = {"0": 100}
    stoploss = -0.20  # 价格口径，纯灾难兜底；常规离场靠 2h 超时（与事件研究口径一致）
    # 止损宽度可注入实验: -0.12 / -0.20
    stoploss = float(os.environ.get("SHOCK_SL", "-0.20"))

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # 触发阈值（BTC 当根跌幅），支持环境变量注入供 walk-forward
    shock_th = float(os.environ.get("BTC_SHOCK_TH", "-0.005"))
    # 只买冲击当根跌幅最深的前 N 个币
    shock_top_n = 3

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
        if current_time - trade.open_date_utc >= HOLD:
            return "hold_timeout"
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # BTC 同周期收益率，按时间戳原样对齐（两边同一时刻收盘，无前视）
        btc = self.dp.get_pair_dataframe(BTC_PAIR, self.timeframe)
        btc_ret = btc.set_index("date")["close"].pct_change().rename("btc_ret")
        dataframe = dataframe.merge(btc_ret.reset_index(), on="date", how="left")

        # 冲击当根各币跌幅的横截面排名（1 = 跌最深）
        rets = {}
        for p in self.config["exchange"]["pair_whitelist"]:
            inf = self.dp.get_pair_dataframe(p, self.timeframe)
            if inf is None or inf.empty:
                continue
            rets[p] = inf.set_index("date")["close"].pct_change()
        wide = pd.DataFrame(rets)
        ranks = wide.rank(axis=1, ascending=True, na_option="keep")  # 值越小 rank 越小
        dataframe = dataframe.merge(
            ranks[[metadata["pair"]]].rename(columns={metadata["pair"]: "shock_rank"}).reset_index(),
            on="date", how="left",
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["btc_ret"] <= self.shock_th)
            # 首波冲击过滤: 前一根 BTC 未处于急跌（避免级联下跌反复接刀）
            & (dataframe["btc_ret"].shift(1) > self.shock_th * 0.5)
            & (dataframe["shock_rank"] <= self.shock_top_n)
            & (dataframe["volume"] > 0),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # BTC 跌势延续且创新低则提前离场（第二波冲击不接）
        dataframe["btc_ret_cum"] = dataframe["btc_ret"].rolling(3).sum()
        dataframe.loc[
            (dataframe["btc_ret_cum"] < -0.02) & (dataframe["volume"] > 0),
            "exit_long"] = 1
        return dataframe

# -*- coding: utf-8 -*-
"""链上聪明钱跟仓策略（freqtrade 桥接版）

信号来源: user_data/onchain_signals.json（由服务器上的 paper_copytrade 写入，
记录监控钱包大额买入的 币对+时间窗）。K 线日期落入任一窗口 -> 下一根开盘入场。
离场: 持有 hold_h 小时（取自信号的 entry_tag）超时。
仅 dry-run 使用——信号文件只向前积累，无历史可回测。
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from pandas import DataFrame

from freqtrade.strategy import IStrategy


class CopyTradeStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short: bool = False

    minimal_roi = {"0": 100}
    stoploss = -0.99  # 跟仓不做价格止损，靠超时离场（模拟阶段观察原始信号质量）

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 5
    default_hold = 12  # 小时

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
        pct = float(os.environ.get("STAKE_PCT", "0.10"))
        stake = self.wallets.get_total_stake_amount() * pct
        if min_stake:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        hold = float((trade.enter_tag or "").replace("h", "") or self.default_hold)
        if current_time - trade.open_date_utc >= timedelta(hours=hold):
            return "hold_timeout"
        return None

    def _signals(self) -> list[dict]:
        f = Path(self.config["user_data_dir"]) / "onchain_signals.json"
        if not f.exists():
            return []
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        import numpy as np
        import pandas as pd
        dataframe["in_window"] = False
        dataframe["sig_hold"] = 0
        sigs = [x for x in self._signals() if x.get("pair") == metadata["pair"]]
        if not sigs:
            return dataframe
        s = pd.DataFrame(sigs)
        s["ws"] = pd.to_datetime(s["window_start"], utc=True)
        s["we"] = pd.to_datetime(s["window_end"], utc=True)
        s = s.sort_values("ws").reset_index(drop=True)
        # 全部换算成 epoch 秒（整数比较，避开 tz-aware dtype 问题）
        candle_t = (dataframe["date"].astype("int64") // 10**9).to_numpy()
        ws = (s["ws"].astype("int64") // 10**9).to_numpy()
        we = (s["we"].astype("int64") // 10**9).to_numpy()
        pos = np.searchsorted(ws, candle_t, side="right") - 1
        valid = pos >= 0
        holds = np.zeros(len(dataframe))
        end_e = np.full(len(dataframe), -1, dtype=np.int64)
        v_idx = np.where(valid)[0]
        if len(v_idx):
            holds[v_idx] = s["hold_h"].to_numpy()[pos[v_idx]]
            end_e[v_idx] = we[pos[v_idx]]
        dataframe["sig_hold"] = holds
        dataframe["in_window"] = valid & (candle_t <= end_e)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_tag"] = None
        mask = dataframe["in_window"].fillna(False).astype(bool) & (dataframe["volume"] > 0)
        dataframe.loc[mask, "enter_tag"] = dataframe.loc[mask, "sig_hold"].astype(int).astype(str) + "h"
        dataframe.loc[mask, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

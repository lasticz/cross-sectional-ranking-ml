# -*- coding: utf-8 -*-
"""内嵌 LightGBM 策略（15m）—— 绕过 FreqAI，直接在策略内训练+预测

训练: 每个币独立训练（简化版，后续可改为跨币训练）
特征: 30 个（价格/成交量/波动率/时段）
目标: 未来 4 根 15m（1h）收益率
入场: 预测 > 阈值做多 / < -阈值做空
"""
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

MODEL_DIR = None  # 延迟初始化


def build_features(df: DataFrame) -> DataFrame:
    """构建 30 个特征（无未来数据）"""
    c = df["close"]
    feat = pd.DataFrame(index=df.index)
    # 价格动量
    for lb in (4, 8, 16, 48, 96, 288):
        feat[f"ret_{lb}"] = c.pct_change(lb)
    # 布林
    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    feat["bb_pos"] = (c - mid) / (2 * sd + 1e-12)
    feat["bb_width"] = (4 * sd) / (mid + 1e-12)
    # RSI
    for p in (7, 14, 28):
        feat[f"rsi_{p}"] = ta.RSI(df, timeperiod=p)
    # EMA 距离
    for sp in (20, 50, 200):
        feat[f"dist_ema{sp}"] = c / c.ewm(span=sp).mean() - 1
    # 成交量
    feat["vol_ratio_4"] = df["volume"] / (df["volume"].rolling(4).mean() + 1e-12)
    feat["vol_ratio_96"] = df["volume"] / (df["volume"].rolling(96).mean() + 1e-12)
    # ATR
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - c.shift(1)), abs(df["low"] - c.shift(1))))
    feat["atr_14"] = tr.rolling(14).mean() / c
    feat["atr_ratio"] = tr.rolling(14).mean() / (tr.rolling(96).mean() + 1e-12)
    # K线形态
    feat["body_ratio"] = np.abs(c - df["open"]) / (df["high"] - df["low"] + 1e-12)
    feat["upper_wick"] = (df["high"] - np.maximum(c, df["open"])) / (df["high"] - df["low"] + 1e-12)
    # 时段
    feat["hour"] = df["date"].dt.hour.to_numpy()
    feat["dow"] = df["date"].dt.dayofweek.to_numpy()
    # 高TF趋势
    feat["htf_trend"] = (c.ewm(span=200).mean() > c.ewm(span=800).mean()).astype(int)
    # 资金费（如果有）
    feat["funding_signal"] = 0  # 占位，后续可注入
    return feat


class InlineMLStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short: bool = False  # 空头在回测中-99.7%, 去掉（与所有研究的空头结论一致）
    minimal_roi = {"0": 100}
    stoploss = -0.90  # 接近无止损(保证金-90%≈价格-30%), 排除止损干扰看纯ML信号
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 400
    max_open_trades = 10  # 扩大并发减少信号拒单

    # 可调参数
    threshold = float(os.environ.get("ML_THRESHOLD", "0.0015"))  # 0.15%
    train_ratio = 0.75  # 前 75% 训练, 后 25% 预测

    # 缓存: pair -> (model, feature_cols)
    _models: dict = {}

    order_types = {
        "entry": "limit", "exit": "limit",
        "stoploss": "market", "stoploss_on_exchange": False,
    }

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return 3.0

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        # 回测中 unlimited stake 会给出极大 proposed_stake, 用固定 25U(≈净值/6)
        return min(proposed_stake, 25.0)

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if current_time - trade.open_date_utc >= timedelta(hours=1):
            return "hold_timeout"
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not HAS_LGB:
            return dataframe
        pair = metadata["pair"]

        feat = build_features(dataframe)
        c = dataframe["close"].to_numpy()
        n = len(c)
        dates = dataframe["date"]

        # 目标: 未来 4 根 15m 收益率
        target = np.full(n, np.nan)
        target[:-4] = c[4:] / c[:-4] - 1

        valid_feat = feat.notna().all(axis=1).to_numpy() & ~np.isnan(target)

        # === Walk-Forward: 每 3 个月重训 ===
        RETRAIN_BARS = 3 * 30 * 96  # ~3 个月的 15m bar 数
        WARMUP_BARS = 365 * 96 // 2  # 至少 6 个月数据才开始训练

        dataframe["ml_pred"] = np.nan

        start = WARMUP_BARS
        while start < n:
            end = min(start + RETRAIN_BARS, n)
            # 训练: 从头到 start
            tr_mask = valid_feat.copy()
            tr_mask[start:] = False  # 只用 start 之前的数据训练
            if tr_mask.sum() < 500:
                start = end
                continue

            X_train = feat[tr_mask]
            y_train = target[tr_mask]

            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, force_col_wise=True
            )
            model.fit(X_train, y_train)

            # 预测: start 到 end（样本外）
            te_mask = valid_feat.copy()
            te_mask[:start] = False
            te_mask[end:] = False
            if te_mask.sum() > 0:
                X_te = feat[te_mask]
                preds = model.predict(X_te)
                dataframe.loc[X_te.index, "ml_pred"] = preds

            start = end

        # 调试输出
        ml = dataframe["ml_pred"].dropna()
        if pair == "ETH/USDT:USDT" and len(ml) > 0:
            print(f"[ML] {pair}: walk-forward预测{len(ml)}根, "
                  f"范围=[{ml.min():.5f},{ml.max():.5f}], "
                  f">0.1%: {(ml>0.001).sum()}, <-0.1%: {(ml<-0.001).sum()}")

        return dataframe

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if "ml_pred" not in dataframe.columns:
            return dataframe
        dataframe.loc[
            (dataframe["ml_pred"] > self.threshold) & (dataframe["volume"] > 0),
            "enter_long"] = 1
        dataframe.loc[
            (dataframe["ml_pred"] < -self.threshold) & (dataframe["volume"] > 0),
            "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

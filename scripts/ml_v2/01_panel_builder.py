# -*- coding: utf-8 -*-
"""
ML v2 - Step 01: 构建横截面面板数据 (panel data)
=================================================

设计理念（与 v1 的根本区别）:
    v1 预测每个币的绝对收益（"SOL 会涨吗"）。
    v2 预测横截面相对收益（"SOL 会比其他 26 个币涨得多吗"），
    天然对冲市场整体方向。

本脚本做四件事:
    1. 从 config.json 的 pair_whitelist 读取币种列表, 加载每个币的
       15m K线 (user_data/data/binance/futures/<BASE>-15m-futures.feather)
       和 8h 资金费率 (<BASE>-1h-funding_rate.feather), 按时间戳对齐拼成
       长表 (panel): 每行 = (时间戳, 币)。
    2. 计算 3 种标签 (4h = 16 根 15m bar 前瞻):
         label_abs      : close.shift(-16)/close - 1        (绝对收益)
         label_neutral  : label_abs - 同一时间戳全市场中位数 (市场中性收益, 主标签)
         label_rank     : label_neutral 在横截面上的百分位 [0,1]
    3. 计算市场层面变量 (每个时间戳一个值, 广播到所有币)。
       注意: 这些是特征列, 全部使用【过去】数据, 不含未来信息:
         market_median_ret_4h       : 全市场过去 4h 收益的横截面中位数
         market_breadth_16          : 过去 16 根 bar (4h) 上涨币的比例 (0-1)
         cross_sectional_dispersion : 当期各币过去 4h 收益的横截面标准差
         btc_ret_4h                 : BTC 过去 4h 收益 (BTC 不在 whitelist,
                                      仅作市场变量; 若无 BTC 数据则回退等权指数)
         btc_vol_96                 : BTC 过去 96 根 bar 15m 对数收益的滚动标准差
    4. 输出 user_data/ml_v2/panel_data.feather 并打印统计。

约定:
    - 标签列使用 shift(-16)（唯一允许的向前看）, 特征列禁止未来数据。
    - 不做异常值清洗（Phase 2 特征工程处理）。
    - 数据已验证 15m 网格无缺口, shift(±16) 与 4h 严格等价。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/ml_v2/ → 项目根
CONFIG_PATH = ROOT / "config.json"
DATA_DIR = ROOT / "user_data" / "data" / "binance" / "futures"
OUT_DIR = ROOT / "user_data" / "ml_v2"
OUT_PATH = OUT_DIR / "panel_data.feather"

TIMEFRAME = "15m"
BARS_4H = 16          # 16 * 15m = 4h 标签/市场变量窗口
VOL_WINDOW = 96       # 96 * 15m = 24h BTC 已实现波动率窗口

FINAL_COLUMNS = [
    "date", "base",
    "open", "high", "low", "close", "volume", "dollar_volume", "funding",
    "label_abs", "label_neutral", "label_rank",
    "market_median_ret_4h", "market_breadth_16",
    "cross_sectional_dispersion", "btc_ret_4h", "btc_vol_96",
]


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def pair_to_base(pair: str) -> str:
    """'ETH/USDT:USDT' -> 'ETH_USDT_USDT' (与 feather 文件名一致)。"""
    return pair.replace("/", "_").replace(":", "_")


def load_klines(base: str) -> pd.DataFrame:
    """加载某币的 15m K线, 返回按 date 排序的 OHLCV。"""
    path = DATA_DIR / f"{base}-{TIMEFRAME}-futures.feather"
    df = pd.read_feather(path)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_funding(base: str) -> pd.Series | None:
    """加载 8h 资金费率 (存在 'open' 列), 返回以 date 为索引的 Series; 无文件返回 None。"""
    path = DATA_DIR / f"{base}-1h-funding_rate.feather"
    if not path.exists():
        return None
    fr = pd.read_feather(path, columns=["date", "open"])
    fr = fr.sort_values("date").drop_duplicates(subset="date")
    return pd.Series(fr["open"].to_numpy(), index=fr["date"].to_numpy(), name="funding")


def attach_funding(klines: pd.DataFrame, base: str) -> pd.DataFrame:
    """把 8h 资金费率前向填充到该币的 15m 时间戳上 (起始前为 NaN)。"""
    fr = load_funding(base)
    if fr is None:
        klines["funding"] = np.nan
        return klines
    klines["funding"] = fr.reindex(klines["date"].to_numpy()).to_numpy()
    klines["funding"] = klines["funding"].ffill()
    return klines


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_panel() -> pd.DataFrame:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    pairs = cfg["exchange"]["pair_whitelist"]
    print(f"[1/4] 加载数据: pair_whitelist 共 {len(pairs)} 个币")

    frames = []
    skipped = []
    for pair in pairs:
        base = pair_to_base(pair)
        kline_path = DATA_DIR / f"{base}-{TIMEFRAME}-futures.feather"
        if not kline_path.exists():
            skipped.append(pair)
            continue
        k = load_klines(base)
        k = attach_funding(k, base)
        k["base"] = base
        frames.append(k)
    if skipped:
        print(f"      跳过 (无 {TIMEFRAME} K线): {skipped}")
    n_loaded = len(frames)

    panel = pd.concat(frames, ignore_index=True)
    # base 用 category 类型: 省 ~40% 字符串内存, groupby 也更快
    panel["base"] = panel["base"].astype("category")
    # 每个币内部按时间排序 (shift 必需), 币之间按 base 排序
    panel = panel.sort_values(["base", "date"], kind="stable").reset_index(drop=True)
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    print(f"      已加载 {n_loaded} 个币, 共 {len(panel):,} 行, "
          f"{panel['date'].min()} -> {panel['date'].max()}")

    # ------------------------------------------------------------------ #
    # [2/4] 标签 (允许使用未来数据 shift(-16), 仅限标签列)
    # ------------------------------------------------------------------ #
    print(f"[2/4] 计算 3 种标签 (前瞻 {BARS_4H} 根 bar = 4h)")
    close_by_base = panel.groupby("base", observed=True)["close"]
    ret_fut = close_by_base.shift(-BARS_4H) / panel["close"] - 1   # 前瞻 4h 收益

    panel["label_abs"] = ret_fut
    # Label B - 市场中性: 同一时间戳上减去全市场中位数
    cs_median = ret_fut.groupby(panel["date"]).transform("median")
    panel["label_neutral"] = ret_fut - cs_median
    # Label C - 横截面百分位 rank
    panel["label_rank"] = (panel["label_neutral"]
                           .groupby(panel["date"]).rank(pct=True))
    del close_by_base, ret_fut, cs_median

    # ------------------------------------------------------------------ #
    # [3/4] 市场层面变量 (严格使用过去数据, 广播到所有币)
    # ------------------------------------------------------------------ #
    print(f"[3/4] 计算市场层面变量 (全部回看, 无未来数据)")
    close_past = panel.groupby("base", observed=True)["close"].shift(BARS_4H)
    ret_past = panel["close"] / close_past - 1                     # 回看 4h 收益
    by_date = ret_past.groupby(panel["date"])
    panel["market_median_ret_4h"] = by_date.transform("median")
    panel["market_breadth_16"] = (ret_past > 0).groupby(panel["date"]).transform("mean")
    panel["cross_sectional_dispersion"] = by_date.transform("std")
    del close_past, ret_past, by_date

    # BTC 市场变量 (BTC 不进面板, 只提供市场信息)
    btc_base = pair_to_base("BTC/USDT:USDT")
    btc_path = DATA_DIR / f"{btc_base}-{TIMEFRAME}-futures.feather"
    if btc_path.exists():
        btc = load_klines(btc_base)
        btc_ret4 = btc["close"] / btc["close"].shift(BARS_4H) - 1
        btc_vol96 = np.log(btc["close"]).diff().rolling(VOL_WINDOW).std()
        idx = btc["date"].to_numpy()
        panel["btc_ret_4h"] = pd.Series(btc_ret4.to_numpy(), index=idx) \
            .reindex(panel["date"].to_numpy()).to_numpy()
        panel["btc_vol_96"] = pd.Series(btc_vol96.to_numpy(), index=idx) \
            .reindex(panel["date"].to_numpy()).to_numpy()
        src = "BTC/USDT:USDT 15m"
    else:
        # 回退: 无 BTC 数据时用全市场中位数收益近似 btc_ret_4h, BTC 波动率置 NaN
        panel["btc_ret_4h"] = panel["market_median_ret_4h"]
        panel["btc_vol_96"] = np.nan
        src = "等权指数回退 (market_median_ret_4h)"
    print(f"      btc_ret_4h / btc_vol_96 来源: {src}")

    panel = panel[FINAL_COLUMNS]
    return panel


# --------------------------------------------------------------------------- #
# 统计打印
# --------------------------------------------------------------------------- #
def print_stats(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("[4/4] 面板统计")
    print("=" * 72)
    n_bases = panel["base"].nunique()
    print(f"总行数        : {len(panel):,}")
    print(f"币数          : {n_bases}")
    print(f"时间范围      : {panel['date'].min()}  ->  {panel['date'].max()}")
    per_base = panel.groupby("base", observed=True).size()
    print(f"每币 bar 数   : min={per_base.min():,} / median={int(per_base.median()):,} "
          f"/ max={per_base.max():,}")

    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    print("\n-- 标签分布 --")
    lab_stats = pd.DataFrame({
        c: panel[c].quantile(qs) for c in ["label_abs", "label_neutral", "label_rank"]
    }).T
    lab_stats.insert(0, "mean", panel[["label_abs", "label_neutral", "label_rank"]].mean())
    lab_stats.insert(1, "std", panel[["label_abs", "label_neutral", "label_rank"]].std())
    lab_stats.columns = ["mean", "std"] + [f"q{int(q*100):02d}" for q in qs]
    print(lab_stats.to_string(float_format=lambda x: f"{x: .6f}"))
    n_tail = int(panel["label_abs"].isna().sum())
    print(f"label NaN 行数: {n_tail:,} (每币最后 {BARS_4H} 根 bar, 属预期)")

    print("\n-- 市场变量分布 --")
    mkt_cols = ["market_median_ret_4h", "market_breadth_16",
                "cross_sectional_dispersion", "btc_ret_4h", "btc_vol_96"]
    mkt_stats = pd.DataFrame({c: panel[c].quantile(qs) for c in mkt_cols}).T
    mkt_stats.insert(0, "mean", panel[mkt_cols].mean())
    mkt_stats.columns = ["mean"] + [f"q{int(q*100):02d}" for q in qs]
    print(mkt_stats.to_string(float_format=lambda x: f"{x: .6f}"))

    print("\n-- 其他列 --")
    print(f"funding  NaN 比例: {panel['funding'].isna().mean():.2%} "
          f"(币种上线早于资金费率数据起点时出现)")
    print(f"dollar_volume 中位数: {panel['dollar_volume'].median():,.0f}")

    # 一致性校验 (必须用【完整】时间戳: 随机抽行会让截面不完整, 中位数不为 0)
    all_dates = panel["date"].drop_duplicates()
    sample_dates = all_dates.sample(min(300, len(all_dates)), random_state=0)
    chk = panel[panel["date"].isin(sample_dates)].dropna(subset=["label_neutral"])
    med_dev = chk.groupby("date", observed=True)["label_neutral"].median().abs().max()
    rank_ok = chk["label_rank"].between(0, 1).all()
    corr = chk["label_neutral"].corr(chk["label_rank"], method="spearman")
    print("\n-- 校验 --")
    print(f"各时间戳 label_neutral 中位数绝对值最大: {med_dev:.2e} (应约等于 0)")
    print(f"label_rank 全部在 [0,1]: {rank_ok}; 与 label_neutral 的 Spearman 相关: {corr:.4f}")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel()

    panel.to_feather(OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n已写出: {OUT_PATH}  ({size_mb:.1f} MB)")

    print_stats(panel)
    print(f"\n总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

# -*- coding: utf-8 -*-
"""ML v2 特征层: panel_data.feather -> 7 个特征族 (~132 列) + 消融测试框架.

输入 (由数据层 01_panel.py 生成):
    user_data/ml_v2/panel_data.feather
    必需列: date, base, open, high, low, close, volume
    可选列: label_abs, label_neutral, label_rank (缺失时按 --label-horizon 补算)

特征族 (FAMILY_ORDER):
    momentum        ret_4/8/16/32/96/288, mom_accel, mom_decay,
                    risk_adj_mom_16, dist_high_96, dist_low_96          (11 raw)
    reversal        ret_1, ret_2, abs_ret_4, max_ret_1, min_ret_1,
                    up_down_ratio_16                                     (6 raw, ret_4 共享)
    volatility      rv_16/96/288, atr_norm_14, range_1h, bb_width_20,
                    vol_of_vol, rv_ratio                                 (8 raw)
    liquidity       log_dollar_volume, volume_ratio_4, volume_shock,
                    amihud_16, amihud_96, volume_trend, turnover_estimate (7 raw)
    btc_relative    ret_16_minus_btc, ret_96_minus_market, btc_beta_96,
                    btc_corr_96, residual_ret_16, residual_mom           (6 raw)
    cross_sectional cs_rank_ret_16/ret_96/volume/rv_16/amihud_96/
                    btc_beta_96/dist_high_96/mom_accel/volume_shock/
                    risk_adj_mom_16                                      (10 列)
    regime          market_breadth_16, cross_sectional_dispersion,
                    btc_ret_16, btc_ret_96, btc_vol_96,
                    hour_sin, hour_cos, dow                              (8 列, 每时间戳一值)

变换 (对 families 1-5 的每个 raw 特征, 共 3 份):
    {name}          raw
    {name}_zscore   时序 z-score: (raw - rolling_mean_96) / (rolling_std_96 + 1e-12),
                    逐币滚动, 只用过去数据
    cs_rank_{name}  横截面百分位: 当期所有币中 rank(pct=True), 每时间戳
                    有效币数 < MIN_CS 时置 NaN
    (命名采用 cs_rank_ 前缀, 与族 6 规格一致)

无未来数据保证:
    - 所有特征仅用 t 及以前的数据 (pct 变化 / rolling, min_periods=完整窗口);
    - 横截面统计只用同一时间戳 t 各币自身的过去信息;
    - 向前看只出现在 label_abs / label_neutral / label_rank (close.shift(-H)).

实现说明 (与规格的三处有意偏差, 均为修正明显问题):
    - amihud_16 / amihud_96 = |ret_1|/dollar_volume 先取逐 bar 非流动性再分别做
      16 / 96 根滚动均值 (规格中 "amihud_96 = rolling_mean_16(amihud_16)" 会与
      amihud_16 重复且无 96 窗口);
    - btc_beta_96 做 [-5, 5] 截断并加 1e-18 方差保护, 避免 BTC 近期方差趋 0 时爆炸;
    - BTC 逐 bar 收益来源按优先级: panel 含 BTC base > 原始 BTC 15m feather
      (--btc-feather, 对齐 panel 时间戳) > 等权市场代理; 滚动 beta/corr 用矩公式
      手写 (pandas 3 的 df.rolling().cov(series) 会生成 T x T 成对矩阵).

消融测试:
    基线 = momentum 族; 按 FAMILY_ORDER 逐族累积; 每轮简化 walk-forward
    (8 个月训练 -> 1 个月预测 -> 滚动), LightGBM(n_estimators=100, max_depth=6,
    learning_rate=0.05); 每个测试时间戳计算预测 vs label_neutral 的 Spearman
    Rank IC, 汇报 mean IC / IC IR / IC t-stat.

输出:
    user_data/ml_v2/feature_matrix.feather   date, base, [特征列], label_abs,
                                            label_neutral, label_rank
    user_data/ml_v2/ablation_results.json    每轮消融的 IC 结果
    user_data/ml_v2/feature_family_map.json  特征名 -> 所属族
    (mock 模式 / panel 缺失时文件名加 _MOCK 后缀, 不污染真实输出)

用法:
    C:/Users/18970/.conda/envs/quant/python.exe scripts/ml_v2/02_feature_families.py
    ... --mock                 # 用模拟 panel 测试逻辑
    ... --no-ablation          # 只建特征矩阵
    ... --train-months 8 --stride 3 --max-folds 99
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path.cwd()  # 项目根（从项目根目录运行: python scripts/ml_v2/xxx.py）
PANEL_PATH = ROOT / "user_data" / "ml_v2" / "panel_data.feather"
OUT_DIR = ROOT / "user_data" / "ml_v2"

EPS = 1e-12
MIN_CS = 5            # 横截面特征: 每时间戳最少有效币数
ZS_WIN = 96           # 时序 z-score 滚动窗口 (96 根 15m bar = 1 天)
REQUIRED_COLS = ("date", "base", "open", "high", "low", "close", "volume")
LABEL_COLS = ("label_abs", "label_neutral", "label_rank")

FAMILY_ORDER = [
    "momentum", "reversal", "volatility", "liquidity",
    "btc_relative", "cross_sectional", "regime",
]

# 族 6: 指定的 10 个 cs_rank 列 -> 归入 cross_sectional 族 (其余 cs_rank 列
# 跟随其 raw 所属族)
CROSS_SECTIONAL_COLS = [
    "cs_rank_ret_16", "cs_rank_ret_96", "cs_rank_volume", "cs_rank_rv_16",
    "cs_rank_amihud_96", "cs_rank_btc_beta_96", "cs_rank_dist_high_96",
    "cs_rank_mom_accel", "cs_rank_volume_shock", "cs_rank_risk_adj_mom_16",
]

LGB_PARAMS = dict(
    n_estimators=100, max_depth=6, learning_rate=0.05,
    verbose=-1, n_jobs=-1, random_state=42,
)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Panel 加载 / 模拟
# --------------------------------------------------------------------------- #
def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_feather(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    return df


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.dropna(subset=["date", "base"])
    df = df.sort_values(["base", "date"]).reset_index(drop=True)
    dup = df.duplicated(subset=["date", "base"]).sum()
    if dup:
        log(f"[WARN] dropping {dup} duplicate (date, base) rows")
        df = df.drop_duplicates(subset=["date", "base"], keep="first")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def make_mock_panel(n_coins: int = 10, months: int = 12, seed: int = 7) -> pd.DataFrame:
    """随机 panel: 市场因子 + 负自相关特质项 (AR(1), phi<0) -> 短期反转可学."""
    rng = np.random.default_rng(seed)
    bars = months * 30 * 96  # 15m bars, 30 天/月
    dates = pd.date_range("2024-01-01", periods=bars, freq="15min", tz="UTC")
    bases = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"][:n_coins]
    mkt = rng.normal(0.0, 0.0035, bars)
    frames = []
    for i, b in enumerate(bases):
        beta = 0.5 + 1.2 * i / max(n_coins - 1, 1)
        phi = -0.18
        e = rng.normal(0.0, 0.006 + 0.004 * i / max(n_coins - 1, 1), bars)
        u = np.empty(bars)
        u[0] = e[0]
        s = np.sqrt(1.0 - phi ** 2)
        for t in range(1, bars):
            u[t] = phi * u[t - 1] + s * e[t]
        r = beta * mkt + u
        close = 100.0 * np.exp(np.cumsum(r))
        open_ = np.empty(bars)
        open_[0], open_[1:] = close[0], close[:-1]
        wig = rng.uniform(0.0002, 0.0015, bars) + 0.25 * np.abs(r)
        high = np.maximum(open_, close) * (1 + wig)
        low = np.minimum(open_, close) * (1 - wig)
        hourfac = 1 + 0.5 * np.sin(np.arange(bars) / 96 * 2 * np.pi)
        vol = np.exp(rng.normal(10 + 0.3 * i, 0.5, bars)) * hourfac * (1 + 8 * np.abs(r))
        frames.append(pd.DataFrame({
            "date": dates, "base": b, "open": open_, "high": high,
            "low": low, "close": close, "volume": vol,
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 宽矩阵工具 (index=date, columns=base)
# --------------------------------------------------------------------------- #
class WideCtx:
    """持有 panel 的宽矩阵与位置索引, 负责 wide -> long 的快速展开."""

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel
        self.w = panel.pivot(index="date", columns="base", values="close")
        self.w = self.w.sort_index().sort_index(axis=1)
        self.dates = self.w.index
        self.bases = list(self.w.columns)
        self.n_ts, self.n_base = self.w.shape
        self._dpos = pd.Index(self.dates).get_indexer(panel["date"].to_numpy())
        self._bpos = pd.Index(self.bases).get_indexer(panel["base"].to_numpy())
        if (self._dpos < 0).any() or (self._bpos < 0).any():
            raise RuntimeError("panel (date, base) not fully covered by pivot")

    def col(self, name: str) -> pd.DataFrame:
        return self.panel.pivot(index="date", columns="base", values=name)\
            .reindex(index=self.dates, columns=self.bases)

    def to_long(self, w: pd.DataFrame) -> np.ndarray:
        """wide (T x N) -> long array 对齐 panel 行顺序 (float32, inf -> NaN)."""
        arr = w.to_numpy(dtype=np.float32)
        out = arr[self._dpos, self._bpos]
        out[~np.isfinite(out)] = np.nan
        return out


def clean(w: pd.DataFrame) -> pd.DataFrame:
    return w.replace([np.inf, -np.inf], np.nan)


def cs_rank_wide(w: pd.DataFrame, min_cs: int) -> pd.DataFrame:
    """横截面百分位 (逐时间戳, 沿 base 轴); 有效币数不足置 NaN."""
    r = w.rank(axis=1, pct=True)
    cnt = w.notna().sum(axis=1).to_numpy()
    bad = cnt < min_cs
    if bad.any():
        r = r.mask(pd.Series(bad, index=w.index), np.nan)
    return r


def transform_wide(name: str, raw: pd.DataFrame, min_cs: int) -> dict:
    """规格 transform() 的宽矩阵版: raw / {name}_zscore / cs_rank_{name}."""
    raw = clean(raw)
    roll = raw.rolling(ZS_WIN, min_periods=ZS_WIN)
    mu = roll.mean()
    sd = roll.std()
    zscore = (raw - mu) / (sd + EPS)
    return {
        name: raw,
        f"{name}_zscore": clean(zscore),
        f"cs_rank_{name}": cs_rank_wide(raw, min_cs),
    }


# --------------------------------------------------------------------------- #
# 特征构建
# --------------------------------------------------------------------------- #
def btc_series(ctx: WideCtx, ret1: pd.DataFrame, btc_feather: Path | None):
    """BTC 逐 bar 收益与窗口收益 (16/96 根). 优先级:
    1. panel 内含 BTC base -> 直接用;
    2. 原始 BTC 15m feather (只读 date/close, 对齐 panel 时间戳, 缺 bar 置 NaN);
    3. 等权市场代理 (panel 各币 ret1 均值).
    """
    if "BTC" in ctx.bases:
        return ret1["BTC"].copy(), None, "panel_base_BTC"
    if btc_feather is not None and btc_feather.exists():
        b = pd.read_feather(btc_feather, columns=["date", "close"])
        b["date"] = b["date"].astype(ctx.dates.dtype)
        b = b.sort_values("date").drop_duplicates("date").set_index("date")
        bc = b["close"].reindex(ctx.dates)  # 未对上的 bar -> NaN
        btc_ret1 = bc / bc.shift(1) - 1.0
        src = f"raw_feather:{btc_feather.name}"
        return btc_ret1, bc, src
    return ret1.mean(axis=1), None, "equal_weight_market_proxy"


def btc_window_ret(btc_ret1: pd.Series, n: int) -> pd.Series:
    logret = np.log1p(btc_ret1.clip(-0.9, 10))
    return np.expm1(logret.rolling(n, min_periods=n).sum())


def build_features(panel: pd.DataFrame, min_cs: int,
                   btc_feather: Path | None = None) -> tuple[dict, dict, dict]:
    """返回 (columns: name->float32 array, family_map: name->family, meta)."""
    ctx = WideCtx(panel)
    log(f"wide matrix: {ctx.n_ts} timestamps x {ctx.n_base} bases")

    c = ctx.col("close")
    h = ctx.col("high")
    l = ctx.col("low")
    v = ctx.col("volume")

    # ---- 基础中间量 ------------------------------------------------------- #
    ret1 = c / c.shift(1) - 1.0
    ret2 = c / c.shift(2) - 1.0
    ret4 = c / c.shift(4) - 1.0
    ret8 = c / c.shift(8) - 1.0
    ret16 = c / c.shift(16) - 1.0
    ret32 = c / c.shift(32) - 1.0
    ret96 = c / c.shift(96) - 1.0
    ret288 = c / c.shift(288) - 1.0
    rv16 = ret1.rolling(16, min_periods=16).std()
    rv96 = ret1.rolling(96, min_periods=96).std()
    rv288 = ret1.rolling(288, min_periods=288).std()
    dollar_vol = c * v

    btc_ret1, _btc_close, btc_src = btc_series(ctx, ret1, btc_feather)
    btc_ret16 = btc_window_ret(btc_ret1, 16)
    btc_ret96 = btc_window_ret(btc_ret1, 96)
    market_median_ret96 = ret96.median(axis=1)

    cols: dict[str, np.ndarray] = {}
    fmap: dict[str, str] = {}

    def add_family(family: str, raws: dict) -> None:
        for name, w in raws.items():
            triple = transform_wide(name, w, min_cs)
            for cname, wv in triple.items():
                if cname in cols:
                    continue  # ret_4 等共享列只算一次
                cols[cname] = ctx.to_long(wv)
                fmap[cname] = family
            del triple
            gc.collect()

    # ---- 族 1: momentum ---------------------------------------------------- #
    add_family("momentum", {
        "ret_4": ret4, "ret_8": ret8, "ret_16": ret16, "ret_32": ret32,
        "ret_96": ret96, "ret_288": ret288,
        "mom_accel": ret4 - ret16,
        "mom_decay": ret96 - ret16,
        "risk_adj_mom_16": ret16 / (rv16 + 1e-8),
        "dist_high_96": c / c.rolling(96, min_periods=96).max() - 1.0,
        "dist_low_96": c / c.rolling(96, min_periods=96).min() - 1.0,
    })
    log(f"momentum done ({len(cols)} cols)")

    # ---- 族 2: reversal ---------------------------------------------------- #
    add_family("reversal", {
        "ret_1": ret1, "ret_2": ret2,
        "abs_ret_4": ret4.abs(),
        "max_ret_1": ret1.rolling(4, min_periods=4).max(),
        "min_ret_1": ret1.rolling(4, min_periods=4).min(),
        "up_down_ratio_16": (ret1 > 0).astype(np.float64).rolling(16, min_periods=16).mean(),
    })
    log(f"reversal done ({len(cols)} cols)")

    # ---- 族 3: volatility -------------------------------------------------- #
    # true range = max(h-l, |h-prev c|, |l-prev c|), 逐元素
    tr = pd.DataFrame(
        np.maximum.reduce([
            (h - l).to_numpy(),
            (h - c.shift(1)).abs().to_numpy(),
            (l - c.shift(1)).abs().to_numpy(),
        ]),
        index=c.index, columns=c.columns,
    )
    rv16_m = rv16.rolling(96, min_periods=96).mean()
    add_family("volatility", {
        "rv_16": rv16, "rv_96": rv96, "rv_288": rv288,
        "atr_norm_14": tr.rolling(14, min_periods=14).mean() / c,
        "range_1h": (h - l) / c,
        "bb_width_20": 4.0 * c.rolling(20, min_periods=20).std() / c.rolling(20, min_periods=20).mean(),
        "vol_of_vol": rv16.rolling(96, min_periods=96).std() / (rv16_m + EPS),
        "rv_ratio": rv16 / (rv96 + EPS),
    })
    log(f"volatility done ({len(cols)} cols)")
    del tr, rv16_m
    gc.collect()

    # ---- 族 4: liquidity --------------------------------------------------- #
    illiq = ret1.abs() / (dollar_vol + 1.0)
    add_family("liquidity", {
        "log_dollar_volume": np.log1p(dollar_vol.clip(lower=0)),
        "volume_ratio_4": v / v.rolling(96, min_periods=96).mean(),
        "volume_shock": v / v.rolling(16, min_periods=16).mean(),
        "amihud_16": illiq.rolling(16, min_periods=16).mean(),
        "amihud_96": illiq.rolling(96, min_periods=96).mean(),
        "volume_trend": v.rolling(16, min_periods=16).mean() / v.rolling(96, min_periods=96).mean(),
        "turnover_estimate": dollar_vol / dollar_vol.rolling(96, min_periods=96).mean(),
    })
    log(f"liquidity done ({len(cols)} cols)")
    del illiq, dollar_vol
    gc.collect()

    # ---- 族 5: btc_relative ------------------------------------------------ #
    # 滚动 beta/corr 用矩公式手写 (df.rolling().cov(series) 在 pandas 3 会生成
    # T x T 成对矩阵而非按列广播, 不可用):
    #   cov = E[xy] - E[x]E[y];  beta = cov / var_btc;  corr = cov / (sd_x sd_btc)
    xy = ret1.mul(btc_ret1, axis=0)
    m_xy = xy.rolling(96, min_periods=96).mean()
    m_x = ret1.rolling(96, min_periods=96).mean()
    m_y = btc_ret1.rolling(96, min_periods=96).mean()
    var_b = btc_ret1.rolling(96, min_periods=96).var()
    sd_x = ret1.rolling(96, min_periods=96).std()
    sd_b = np.sqrt(var_b.clip(lower=0.0))
    cov = m_xy - m_x.mul(m_y, axis=0)
    beta = (cov.div(var_b + 1e-18, axis=0)).clip(-5.0, 5.0)
    corr = cov.div(sd_x.mul(sd_b, axis=0) + EPS, axis=0).clip(-1.0, 1.0)
    resid16 = ret16 - beta.mul(btc_ret16, axis=0)
    add_family("btc_relative", {
        "ret_16_minus_btc": ret16.sub(btc_ret16, axis=0),
        "ret_96_minus_market": ret96.sub(market_median_ret96, axis=0),
        "btc_beta_96": beta,
        "btc_corr_96": corr,
        "residual_ret_16": resid16,
        "residual_mom": resid16.rolling(96, min_periods=96).mean(),
    })
    log(f"btc_relative done ({len(cols)} cols; btc source = {btc_src})")
    del xy, m_xy, m_x, m_y, var_b, sd_x, sd_b, cov, beta, corr, resid16
    gc.collect()

    # ---- 族 6: cross_sectional (10 个指定 cs_rank 列; 9 个已存在) ---------- #
    if "cs_rank_volume" not in cols:
        wv = cs_rank_wide(clean(v), min_cs)
        cols["cs_rank_volume"] = ctx.to_long(wv)
        fmap["cs_rank_volume"] = "cross_sectional"
    missing_cs = [c0 for c0 in CROSS_SECTIONAL_COLS if c0 not in cols]
    if missing_cs:
        raise RuntimeError(f"cross-sectional columns missing: {missing_cs}")
    for c0 in CROSS_SECTIONAL_COLS:  # 族归属重定向到 cross_sectional
        fmap[c0] = "cross_sectional"
    log(f"cross_sectional done ({len(cols)} cols)")

    # ---- 族 7: regime (每时间戳一个值, 广播到所有币) ------------------------ #
    hour = ctx.dates.hour.to_numpy(dtype=np.float64)
    dow = ctx.dates.dayofweek.to_numpy(dtype=np.float64)
    btc_vol96 = btc_ret1.rolling(96, min_periods=96).std()
    valid16 = ret16.notna().sum(axis=1)
    regime = {
        "market_breadth_16": (ret16 > 0).sum(axis=1) / valid16.replace(0, np.nan),
        "cross_sectional_dispersion": ret16.std(axis=1),
        "btc_ret_16": btc_ret16,
        "btc_ret_96": btc_ret96,
        "btc_vol_96": btc_vol96,
        "hour_sin": pd.Series(np.sin(2 * np.pi * hour / 24.0), index=ctx.dates),
        "hour_cos": pd.Series(np.cos(2 * np.pi * hour / 24.0), index=ctx.dates),
        "dow": pd.Series(dow, index=ctx.dates),
    }
    # 每时间戳一个值: 直接按 date 位置索引广播到 panel 各行
    for name, s in regime.items():
        arr = s.reindex(ctx.dates).to_numpy(dtype=np.float64)
        vals = arr[ctx._dpos].astype(np.float32)
        vals[~np.isfinite(vals)] = np.nan
        cols[name] = vals
        fmap[name] = "regime"
    log(f"regime done ({len(cols)} cols)")

    meta = {
        "n_timestamps": int(ctx.n_ts), "n_bases": int(ctx.n_base),
        "bases": ctx.bases, "btc_source": btc_src,
        "zs_window": ZS_WIN, "min_cs_count": min_cs,
        "n_features": len(cols),
    }
    return cols, fmap, meta


# --------------------------------------------------------------------------- #
# 标签 (仅当 panel 未提供时补算)
# --------------------------------------------------------------------------- #
def ensure_labels(panel: pd.DataFrame, horizon: int, min_cs: int) -> pd.DataFrame:
    if all(c0 in panel.columns for c0 in LABEL_COLS):
        log("labels found in panel, using as-is")
        return panel
    log(f"labels missing -> computing from close, horizon={horizon} bars")
    ctx = WideCtx(panel)
    c = ctx.col("close")
    fwd = c.shift(-horizon) / c - 1.0  # 全脚本唯一"向前看"
    label_abs = ctx.to_long(fwd)

    # label_neutral: 减去当期横截面均值; label_rank: 当期百分位
    fwd_c = fwd.copy()
    cnt = fwd_c.notna().sum(axis=1)
    demean = fwd_c.sub(fwd_c.mean(axis=1), axis=0)
    demean = demean.mask(pd.Series(cnt.to_numpy() < min_cs, index=fwd_c.index), np.nan)
    label_neutral = ctx.to_long(demean)
    label_rank = ctx.to_long(cs_rank_wide(fwd_c, min_cs))

    panel = panel.copy()
    panel["label_abs"] = label_abs
    panel["label_neutral"] = label_neutral
    panel["label_rank"] = label_rank
    return panel


# --------------------------------------------------------------------------- #
# 消融测试
# --------------------------------------------------------------------------- #
def per_ts_spearman(dcodes: np.ndarray, x: np.ndarray, y: np.ndarray,
                    min_n: int) -> tuple[np.ndarray, np.ndarray]:
    """逐时间戳 Spearman (秩相关). 返回 (dcode, corr), 只保留 n>=min_n 且方差>0."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return np.array([]), np.array([])
    d = dcodes[m]
    g = pd.Series(d)
    ra = pd.Series(pd.Series(x[m]).groupby(g).rank()).reset_index(drop=True)
    rb = pd.Series(pd.Series(y[m]).groupby(g).rank()).reset_index(drop=True)
    g = g.reset_index(drop=True)
    n = ra.groupby(g).size()
    ma = ra.groupby(g).mean()
    mb = rb.groupby(g).mean()
    mab = (ra * rb).groupby(g).mean()
    va = (ra * ra).groupby(g).mean()
    vb = (rb * rb).groupby(g).mean()
    cov = mab - ma * mb
    den = np.sqrt((va - ma ** 2) * (vb - mb ** 2))
    corr = cov / den.replace(0.0, np.nan)
    ok = (n >= min_n) & np.isfinite(corr)
    return n.index.to_numpy()[ok], corr.to_numpy()[ok]


def walk_forward_ic(X: np.ndarray, y: np.ndarray, month: np.ndarray, dcode: np.ndarray,
                    feat_names: list[str], train_months: int, stride: int,
                    max_folds: int, min_ic_coins: int) -> dict:
    """简化 walk-forward: 8 个月训练 -> 1 个月预测 -> 滚动. 返回 IC 统计."""
    import lightgbm as lgb

    months = np.unique(month)
    months.sort()
    n_folds = max(0, len(months) - train_months)
    if max_folds > 0:
        n_folds = min(n_folds, max_folds)
    if n_folds <= 0:
        return {"error": f"not enough months for walk-forward "
                         f"({len(months)} available, need > {train_months})"}

    pred = np.full(len(y), np.nan, dtype=np.float32)
    valid_y = np.isfinite(y)
    fold_month_labels = []
    fold_ics = []

    for i in range(train_months, train_months + n_folds):
        m_test = months[i]
        te = (month == m_test) & valid_y
        tr_months = months[i - train_months:i]
        tr = np.isin(month, tr_months) & valid_y
        if stride > 1:  # 训练时间戳降采样 (保留完整横截面)
            tr_dates = np.unique(dcode[tr])
            keep = tr_dates[::stride]
            tr = tr & np.isin(dcode, keep)
        if tr.sum() < 500 or te.sum() < 100:
            fold_month_labels.append(int(m_test))
            fold_ics.append(float("nan"))
            continue
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X[tr], y[tr])
        # 测试行剔除特征几乎全 NaN 的早期 bar
        nan_frac = np.isnan(X[te]).mean(axis=1)
        ok_te = nan_frac < 0.9
        te_idx = np.where(te)[0][ok_te]
        pred[te_idx] = model.predict(X[te_idx])
        # 该 fold 的月度 IC = fold 内逐时间戳 IC 的均值
        _, corr_f = per_ts_spearman(dcode[te_idx], pred[te_idx], y[te_idx], min_ic_coins)
        fold_month_labels.append(int(m_test))
        fold_ics.append(float(np.nanmean(corr_f)) if len(corr_f) else float("nan"))

    m = np.isfinite(pred) & valid_y
    d, corr = per_ts_spearman(dcode[m], pred[m], y[m], min_ic_coins)
    if len(corr) < 10:
        return {"error": "too few valid timestamps for IC"}
    mean_ic = float(np.nanmean(corr))
    std_ic = float(np.nanstd(corr, ddof=1))
    return {
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "ic_ir": float(mean_ic / std_ic) if std_ic > 0 else 0.0,
        "ic_tstat": float(mean_ic / std_ic * np.sqrt(len(corr))) if std_ic > 0 else 0.0,
        "n_ts": int(len(corr)),
        "n_pred_rows": int(m.sum()),
        "n_folds": int(n_folds),
        "fold_month": fold_month_labels,
        "fold_ic": fold_ics,
    }


def ablation_test(src, family_map: dict[str, str],
                  label_col: str = "label_neutral", train_months: int = 8,
                  stride: int = 3, max_folds: int = 0, min_ic_coins: int = 10) -> dict:
    """逐族添加特征, 每轮一次快速 walk-forward, 记录 OOS Rank IC.

    src: feature_matrix DataFrame, 或其 feather 路径 (每轮只读当轮所需列,
    控制内存: 400 万行 x ~123 列 float32 全量驻留 ~2GB).
    """
    import lightgbm as lgb

    from_path = isinstance(src, (str, Path))
    feature_cols = [c0 for c0 in family_map]  # 输出矩阵中按 family_map 键存在
    if from_path:
        base = pd.read_feather(src, columns=["date", label_col])
    else:
        base = src[["date", label_col]].copy()
    y = base[label_col].to_numpy(dtype=np.float64)
    dcode = pd.factorize(base["date"])[0].astype(np.int32)
    month = (base["date"].dt.year * 12 + base["date"].dt.month).to_numpy(np.int32)
    del base
    gc.collect()

    by_family: dict[str, list[str]] = {f: [] for f in FAMILY_ORDER}
    for c0 in feature_cols:
        by_family[family_map[c0]].append(c0)

    results = {"rounds": [], "families_present": {k: len(v) for k, v in by_family.items()}}
    for k, fam in enumerate(FAMILY_ORDER):
        fams = FAMILY_ORDER[:k + 1]
        cols = sorted({c0 for f in fams for c0 in by_family[f]})
        t0 = time.time()
        if from_path:
            X = pd.read_feather(src, columns=cols).to_numpy(dtype=np.float32)
        else:
            X = src[cols].to_numpy(dtype=np.float32)
        r = walk_forward_ic(X, y, month, dcode, cols, train_months, stride,
                            max_folds, min_ic_coins)
        del X
        gc.collect()
        if "error" in r:
            log(f"  [ablation] {fams} -> ERROR: {r['error']}")
            results["rounds"].append({"added_family": fam, "families": fams,
                                      "n_features": len(cols), **r})
            break
        results["rounds"].append({"added_family": fam, "families": fams,
                                  "n_features": len(cols), **r})
        log(f"  [ablation] +{fam:<16s} feats={len(cols):>3d} "
            f"IC={r['mean_ic']:+.4f} IR={r['ic_ir']:+.3f} "
            f"t={r['ic_tstat']:+.2f} ({time.time() - t0:.0f}s)")
    return results


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=PANEL_PATH)
    p.add_argument("--btc-feather", type=Path,
                   default=ROOT / "user_data" / "data" / "binance" / "futures" /
                   "BTC_USDT_USDT-15m-futures.feather",
                   help="BTC 15m feather used for family 5/7 when panel has no BTC base")
    p.add_argument("--mock", action="store_true", help="force mock panel")
    p.add_argument("--mock-coins", type=int, default=10)
    p.add_argument("--mock-months", type=int, default=12)
    p.add_argument("--label-horizon", type=int, default=16,
                   help="bars for fallback label computation (4h @15m)")
    p.add_argument("--label-col", default="label_neutral")
    p.add_argument("--min-cs", type=int, default=MIN_CS)
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--train-months", type=int, default=8)
    p.add_argument("--stride", type=int, default=3,
                   help="training timestamp stride (1 = use all)")
    p.add_argument("--max-folds", type=int, default=0, help="0 = all folds")
    p.add_argument("--ic-min-coins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mock = args.mock or not args.panel.exists()
    if mock:
        if not args.mock and not args.panel.exists():
            log(f"!! panel not found at {args.panel} -> RUNNING ON MOCK DATA, "
                f"outputs get _MOCK suffix")
        panel = make_mock_panel(args.mock_coins, args.mock_months, args.seed)
        log(f"mock panel: {len(panel):,} rows, {panel['base'].nunique()} coins, "
            f"{panel['date'].min()} -> {panel['date'].max()}")
        suffix = "_MOCK"
    else:
        panel = load_panel(args.panel)
        suffix = ""

    panel = prepare_panel(panel)
    panel = ensure_labels(panel, args.label_horizon, args.min_cs)

    cols, family_map, meta = build_features(panel, args.min_cs, args.btc_feather)

    # ---- 列顺序 (按族排列) -------------------------------------------------- #
    order = [c0 for f in FAMILY_ORDER for c0 in
             sorted([k for k, ff in family_map.items() if ff == f])]

    # ---- 校验 (写盘前, 基于列 dict) ----------------------------------------- #
    nan_rates = sorted(((k, float(np.isnan(cols[k]).mean())) for k in order),
                       key=lambda kv: -kv[1])
    log("top-8 NaN features: " + ", ".join(f"{k}={v:.1%}" for k, v in nan_rates[:8]))
    all_nan = [k for k in order if np.isnan(cols[k]).all()]
    if all_nan:
        log(f"[WARN] all-NaN features (kept but flagged): {all_nan}")
    rk = pd.Series(cols["cs_rank_ret_16"]).groupby(panel["date"].to_numpy()).mean()
    log(f"sanity: cs_rank_ret_16 per-ts mean median={rk.median():.3f} (expect ~(n+1)/2n), "
        f"label_neutral NaN={panel['label_neutral'].isna().mean():.1%}")

    # ---- 写 feature_matrix (pyarrow 列式, 避免 pandas 块合并的双份驻留) ------ #
    import pyarrow as pa
    import pyarrow.feather as paf

    out_feather = OUT_DIR / f"feature_matrix{suffix}.feather"
    arrays = [pa.array(panel["date"]), pa.array(panel["base"])]
    arrays += [pa.array(cols[k]) for k in order]
    arrays += [pa.array(panel[c0].to_numpy(dtype=np.float32)) for c0 in LABEL_COLS]
    tbl = pa.Table.from_arrays(arrays, names=["date", "base", *order, *LABEL_COLS])
    paf.write_feather(tbl, out_feather)
    n_rows = tbl.num_rows
    del arrays, tbl, cols
    gc.collect()
    log(f"saved {out_feather.name}: {n_rows:,} rows x {2 + len(order) + len(LABEL_COLS)} cols "
        f"({out_feather.stat().st_size / 1e6:.1f} MB)")

    # ---- 特征 -> 族映射 ----------------------------------------------------- #
    fm_path = OUT_DIR / f"feature_family_map{suffix}.json"
    fam_lists = {f: sorted([k for k, ff in family_map.items() if ff == f])
                 for f in FAMILY_ORDER}
    fm_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_to_family": family_map,
        "families": fam_lists,
        "meta": meta,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved {fm_path.name}: {len(family_map)} features across {len(fam_lists)} families")

    # ---- 消融 -------------------------------------------------------------- #
    if not args.no_ablation:
        log(f"ablation: walk-forward train={args.train_months}m, stride={args.stride}, "
            f"lgb={ {k: v for k, v in LGB_PARAMS.items() if k != 'verbose'} }")
        res = ablation_test(out_feather, family_map, args.label_col,
                            args.train_months, args.stride, args.max_folds,
                            args.ic_min_coins)
        abl_path = OUT_DIR / f"ablation_results{suffix}.json"
        abl_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target": args.label_col,
            "walk_forward": {"train_months": args.train_months,
                             "test_months": 1, "train_stride": args.stride,
                             "max_folds": args.max_folds},
            "lgbm_params": LGB_PARAMS,
            "is_mock": mock,
            **res,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"saved {abl_path.name}")

        print("\n===== ABLATION SUMMARY (cumulative families -> OOS Rank IC) =====")
        print(f"{'round':<8s}{'added family':<18s}{'#feat':>6s}{'mean IC':>10s}"
              f"{'IC IR':>9s}{'t-stat':>8s}{'#ts':>7s}")
        for i, r in enumerate(res.get("rounds", [])):
            if "error" in r:
                print(f"{i:<8d}{r['added_family']:<18s}{r['n_features']:>6d}"
                      f"{'ERROR':>10s}")
                continue
            print(f"{i:<8d}{r['added_family']:<18s}{r['n_features']:>6d}"
                  f"{r['mean_ic']:>10.4f}{r['ic_ir']:>9.3f}"
                  f"{r['ic_tstat']:>8.2f}{r['n_ts']:>7d}")

    log(f"done in {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

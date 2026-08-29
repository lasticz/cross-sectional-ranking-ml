# -*- coding: utf-8 -*-
"""ML 特征工厂 (特征层): clean_data.feather -> 104 候选特征 -> 分段 Spearman IC 筛选
-> 层次聚类去冗余 -> 50-80 个精选特征 + 目标列.

输入 (由 data_pipeline.py 生成):
    user_data/ml/clean_data.feather
    列: date, base, open, high, low, close, volume, funding, _cs_*(横截面 z)

流程
----
1. 逐币生成 ~93 个时序候选特征 (动量/波动率/成交量/技术指标/布林/K线形态/
   时段/资金费), 全部只用过去数据 (rolling/ewm 均 min_periods=完整窗口).
2. 在 (date x base) 宽矩阵上计算 11 个横截面特征 (rank / zscore / demean,
   每个时间戳至少 5 个币才有效).
3. 目标变量: target_4bar / target_16bar = 未来 4/16 根 15m bar 简单收益率
   (全脚本唯一"向前看"的地方, 用 numpy 数组切片实现, 等价 shift(-N)).
4. IC 分析 (约 50 万行抽样, 按时间戳 stride 采样保留完整横截面):
   时间等分 20 段; 每段内计算逐时间戳横截面 Spearman IC 再平均;
   日历类特征 (横截面内无差异) 退化为段内 pooled Spearman.
   段均值/标准差 -> t-stat = mean/std*sqrt(n_seg), IC IR = mean/std.
5. 筛选: |IC均值| > 0.008 且 |t-stat| > 1.5 (任一 horizon 通过即可).
6. 去冗余: 候选间 Spearman 相关 (秩相关), 层次聚类 (average linkage,
   距离 = 1-|corr|, 阈值 |corr|>0.7 同簇), 每簇保留分数 (|IC IR|) 最大者;
   不足 50 个时按分数贪心回填 (优先与新选中特征 |corr|<=0.7 者),
   超过 80 个时按分数截断.
7. 输出:
    user_data/ml/feature_matrix.feather  date, base, [选中特征], target_4bar, target_16bar
    user_data/ml/feature_report.json     每个候选特征的 IC / 排名 / 是否选中

无未来数据保证:
    - 所有特征仅用 t 及以前的数据 (diff / rolling / ewm / 正向 shift);
    - 横截面统计只用同一时间戳 t 各币自身的过去信息;
    - 向前看只出现在 target_4bar / target_16bar 两列.

用法:
    C:/Users/18970/.conda/envs/quant/python.exe scripts/ml/feature_factory.py
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

ROOT = Path.cwd()  # 项目根（从项目根目录运行: python scripts/ml/xxx.py）
IN_FEATHER = ROOT / "user_data" / "ml" / "clean_data.feather"
OUT_DIR = ROOT / "user_data" / "ml"
OUT_FEATHER = OUT_DIR / "feature_matrix.feather"
OUT_JSON = OUT_DIR / "feature_report.json"

EPS = 1e-12
MIN_CS_COUNT = 5            # 横截面特征: 每个时间戳最少币数
MIN_IC_OBS_PER_SEG = 50     # 每段最少有效逐时间戳 IC 数
MIN_SEG_FOR_STATS = 10      # 汇总 IC 统计所需最少有效段数

RET_WINDOWS = (2, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 160, 192, 240, 288)
EMA_WINDOWS = (20, 50, 100, 200)
ATR_WINDOWS = (7, 14, 48, 96)
BB_WINDOWS = (20, 50, 100)
RV_WINDOWS = (16, 48, 96, 320)
RSI_WINDOWS = (7, 10, 14, 21, 28, 42)
VOLR_WINDOWS = (4, 12, 48, 96, 192)

# 横截面内取常数的特征 -> IC 用段内 pooled Spearman
CALENDAR_FEATURES = ("hour_sin", "hour_cos", "is_us_session", "is_asia_session", "dow")

CS_RANK_SRC = (             # (源列, 输出名)
    ("ret_96", "cs_rank_ret_96"),
    ("ret_288", "cs_rank_ret_288"),
    ("rsi_14", "cs_rank_rsi_14"),
    ("vol_ratio_96", "cs_rank_vol_ratio_96"),
    ("dist_ema50", "cs_rank_dist_ema50"),
    ("funding_level", "cs_rank_funding"),
)
CS_Z_SRC = (
    ("ret_96", "cs_zscore_ret_96"),
    ("ret_16", "cs_zscore_ret_16"),
)
CS_DEMEAN_SRC = (("ret_16", "cs_dist_from_mean"),)
CS_PASSTHRU = (("_cs_volume", "cs_volume_z"), ("_cs_funding", "cs_funding_z"))
CS_FEATURE_NAMES = (
    [n for _, n in CS_RANK_SRC]
    + [n for _, n in CS_Z_SRC]
    + [n for _, n in CS_DEMEAN_SRC]
    + [n for _, n in CS_PASSTHRU]
)
TARGET_COLS = ("target_4bar", "target_16bar")

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 逐币时序特征
# --------------------------------------------------------------------------- #
def per_coin_features(g: pd.DataFrame) -> pd.DataFrame:
    """对一个币的连续 15m 序列计算全部时序候选特征 (只用过去数据)."""
    idx = g.index
    o = g["open"].to_numpy(np.float64)
    h = g["high"].to_numpy(np.float64)
    l = g["low"].to_numpy(np.float64)
    c = g["close"].to_numpy(np.float64)
    v = g["volume"].to_numpy(np.float64)
    f = g["funding"].to_numpy(np.float64)

    cs = pd.Series(c, index=idx)                    # close
    hs = pd.Series(h, index=idx)
    ls = pd.Series(l, index=idx)
    vs = pd.Series(v, index=idx)
    fs = pd.Series(f, index=idx)
    prev_c = cs.shift(1)
    logc = pd.Series(np.log(np.where(c > 0, c, np.nan)), index=idx)
    r1 = logc.diff()
    out: dict[str, np.ndarray] = {}

    # ---- 动量 ----
    for w in RET_WINDOWS:
        out[f"ret_{w}"] = logc.diff(w).to_numpy()
    emas = {w: cs.ewm(span=w, adjust=False, min_periods=w).mean() for w in EMA_WINDOWS}
    for w in EMA_WINDOWS:
        out[f"dist_ema{w}"] = (cs / emas[w] - 1.0).to_numpy()
    out["ema_ratio"] = (emas[20] / emas[50] - 1.0).to_numpy()
    d16 = logc.diff(16)
    out["momentum_accel"] = (d16 - d16.shift(16)).to_numpy()
    d4 = logc.diff(4)
    out["momentum_accel_4"] = (d4 - d4.shift(4)).to_numpy()
    out["dist_high_288"] = (cs / hs.rolling(288, min_periods=288).max() - 1.0).to_numpy()
    out["dist_low_288"] = (cs / ls.rolling(288, min_periods=288).min() - 1.0).to_numpy()
    for w in (16, 48, 96):
        num = (cs - cs.shift(w)).abs()
        den = r1.abs().rolling(w, min_periods=w).sum() + EPS
        out[f"er_{w}"] = (num / den).to_numpy()
    out["ret_autocorr_96"] = r1.rolling(96, min_periods=48).corr(r1.shift(1)).to_numpy()

    # ---- 波动率 ----
    tr = pd.concat([(hs - ls), (hs - prev_c).abs(), (ls - prev_c).abs()], axis=1).max(axis=1)
    for w in ATR_WINDOWS:
        out[f"atr_norm_{w}"] = (tr.rolling(w, min_periods=w).mean() / cs).to_numpy()
    for w in BB_WINDOWS:
        mid = cs.rolling(w, min_periods=w).mean()
        sd = cs.rolling(w, min_periods=w).std()
        out[f"bb_width_{w}"] = (4.0 * sd / mid).to_numpy()
        out[f"bb_pos_{w}"] = ((cs - mid) / (2.0 * sd + EPS)).to_numpy()
    rv = {}
    for w in RV_WINDOWS:
        rv[w] = r1.rolling(w, min_periods=w).std()
        out[f"realized_vol_{w}"] = rv[w].to_numpy()
    out["vol_of_vol"] = rv[16].rolling(48, min_periods=48).std().to_numpy()
    out["vol_of_vol_48"] = rv[48].rolling(96, min_periods=96).std().to_numpy()
    out["vol_curve"] = (rv[16] - rv[48]).to_numpy()
    out["ret_abs_mean_48"] = r1.abs().rolling(48, min_periods=48).mean().to_numpy()
    bw20 = pd.Series(out["bb_width_20"], index=idx)
    out["bb_squeeze"] = (bw20 / (bw20.rolling(96, min_periods=96).mean() + EPS)).to_numpy()
    bw50 = pd.Series(out["bb_width_50"], index=idx)
    out["bb_squeeze_50"] = (bw50 / (bw50.rolling(96, min_periods=96).mean() + EPS)).to_numpy()
    hh48, ll48 = hs.rolling(48, min_periods=48).max(), ls.rolling(48, min_periods=48).min()
    out["range_pos_48"] = ((cs - ll48) / (hh48 - ll48 + EPS)).clip(0.0, 1.0).to_numpy()
    hh16, ll16 = hs.rolling(16, min_periods=16).max(), ls.rolling(16, min_periods=16).min()
    out["close_pos_16"] = ((cs - ll16) / (hh16 - ll16 + EPS)).clip(0.0, 1.0).to_numpy()

    # ---- 成交量 ----
    logv = pd.Series(np.log1p(np.where(v > 0, v, np.nan)), index=idx)
    for w in VOLR_WINDOWS:
        out[f"vol_ratio_{w}"] = (vs / (vs.rolling(w, min_periods=w).mean() + EPS)).to_numpy()
    out["vol_trend"] = (
        vs.rolling(4, min_periods=4).mean() / (vs.rolling(96, min_periods=96).mean() + EPS) - 1.0
    ).to_numpy()
    out["volume_ma_ratio"] = (
        vs.rolling(4, min_periods=4).mean() / (vs.rolling(48, min_periods=48).mean() + EPS) - 1.0
    ).to_numpy()
    out["volume_price_corr_20"] = logv.rolling(20, min_periods=12).corr(r1).to_numpy()
    out["volume_price_corr_48"] = logv.rolling(48, min_periods=24).corr(r1).to_numpy()
    lm = logv.rolling(96, min_periods=96)
    out["log_volume_z"] = ((logv - lm.mean()) / (lm.std() + EPS)).to_numpy()

    # ---- 技术指标 ----
    up_move = r1.clip(lower=0.0)
    dn_move = (-r1).clip(lower=0.0)
    for w in RSI_WINDOWS:
        au = up_move.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
        ad = dn_move.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
        out[f"rsi_{w}"] = (100.0 - 100.0 / (1.0 + au / (ad + EPS))).to_numpy()
    ema12 = cs.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = cs.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = ((macd - sig) / cs).to_numpy()
    for w, name in ((14, "stoch_k"), (48, "stoch_k_48")):
        hh = hs.rolling(w, min_periods=w).max()
        ll = ls.rolling(w, min_periods=w).min()
        # 注: 数据层 ffill 修复会留下少量 close 越出 high/low 的 bar,
        # clip 回自然区间, 避免 oscillator 出现伪极值.
        k = ((cs - ll) / (hh - ll + EPS)).clip(0.0, 1.0)
        out[name] = k.to_numpy()
        if w == 14:
            out["stoch_d_14"] = k.rolling(3, min_periods=3).mean().to_numpy()
            out["williams_r"] = (-(hh - cs) / (hh - ll + EPS) * 100.0).clip(-100.0, 0.0).to_numpy()
    tp = (hs + ls + cs) / 3.0
    sma_tp = tp.rolling(20, min_periods=20).mean()
    mad = (tp - sma_tp).abs().rolling(20, min_periods=20).mean()
    out["cci_20"] = ((tp - sma_tp) / (0.015 * mad + EPS)).to_numpy()
    mf = tp * vs
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp < tp.shift(1), 0.0)
    for w in (7, 14):
        ps = pos_mf.rolling(w, min_periods=w).sum()
        ns = neg_mf.rolling(w, min_periods=w).sum()
        out[f"mfi_{w}"] = (100.0 * ps / (ps + ns + EPS)).to_numpy()

    # ---- K 线形态 ----
    rng = h - l
    denom = np.where(rng > 0, rng, np.nan)
    body = pd.Series(np.clip((c - o) / denom, -1.0, 1.0), index=idx)
    out["body_ratio"] = body.to_numpy()
    uw = pd.Series(np.clip((h - np.maximum(o, c)) / denom, 0.0, 1.0), index=idx)
    lw = pd.Series(np.clip((np.minimum(o, c) - l) / denom, 0.0, 1.0), index=idx)
    out["upper_wick"] = uw.to_numpy()
    out["lower_wick"] = lw.to_numpy()
    out["wick_diff"] = (uw - lw).to_numpy()
    out["body_ratio_ma_16"] = body.rolling(16, min_periods=16).mean().to_numpy()
    up = (r1 > 0).fillna(False)
    dn = (r1 < 0).fillna(False)
    grp_u = (up != up.shift(1)).cumsum()
    out["consecutive_up"] = up.astype(np.int32).groupby(grp_u).cumsum().clip(upper=50).to_numpy()
    grp_d = (dn != dn.shift(1)).cumsum()
    out["consecutive_down"] = dn.astype(np.int32).groupby(grp_d).cumsum().clip(upper=50).to_numpy()
    out["up_ratio_16"] = up.rolling(16, min_periods=16).mean().to_numpy()
    out["up_ratio_48"] = up.rolling(48, min_periods=48).mean().to_numpy()

    # ---- 资金费 ----
    out["funding_level"] = fs.to_numpy()
    out["funding_ma_3"] = fs.rolling(3, min_periods=3).mean().to_numpy()
    out["funding_diff_3"] = (fs - fs.rolling(3, min_periods=3).mean()).to_numpy()
    fm = fs.rolling(96, min_periods=96)
    out["funding_extreme"] = ((fs - fm.mean()) / (fm.std() + EPS)).to_numpy()
    out["funding_sum_12"] = fs.rolling(12, min_periods=12).sum().to_numpy()

    # ---- 时段 ----
    dt = g["date"].dt
    hr = dt.hour + dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hr / 24.0).to_numpy()
    out["hour_cos"] = np.cos(2.0 * np.pi * hr / 24.0).to_numpy()
    out["is_us_session"] = ((hr >= 13) & (hr < 21)).astype(np.float64).to_numpy()
    out["is_asia_session"] = (hr < 8).astype(np.float64).to_numpy()
    out["dow"] = dt.dayofweek.to_numpy(np.float64)

    # ---- 目标 (唯一向前看: 等价 shift(-4) / shift(-16)) ----
    n = len(c)
    t4 = np.full(n, np.nan)
    t16 = np.full(n, np.nan)
    t4[:-4] = c[4:] / c[:-4] - 1.0
    t16[:-16] = c[16:] / c[:-16] - 1.0
    out["target_4bar"] = t4
    out["target_16bar"] = t16

    return pd.DataFrame(out, index=idx).astype(np.float32)


# --------------------------------------------------------------------------- #
# 横截面特征 (date x base 宽矩阵)
# --------------------------------------------------------------------------- #
def wide_scatter(vals: np.ndarray, dc: np.ndarray, bc: np.ndarray, nD: int, nB: int) -> np.ndarray:
    W = np.full((nD, nB), np.nan, np.float64)
    W[dc, bc] = vals
    return W


def add_cross_sectional(
    MAT: np.ndarray,
    col_idx: dict[str, int],
    dc: np.ndarray,
    bc: np.ndarray,
    nD: int,
    nB: int,
    passthrough: dict[str, np.ndarray],
) -> None:
    """就地填充 MAT 预留的横截面特征列.

    dc/bc 为行序下的 factorize 编码; 宽矩阵变换后用 (dc, bc) 花式索引散射回长表,
    对不完整网格 (部分币种晚上市/缺 bar) 也精确对齐.
    """

    def cnt_of(Wa: np.ndarray) -> np.ndarray:
        return np.sum(~np.isnan(Wa), axis=1)

    def rank_pct(Wa: np.ndarray) -> np.ndarray:
        cnt = cnt_of(Wa)
        with np.errstate(invalid="ignore"):
            R = rankdata(Wa, axis=1, nan_policy="omit")
            R = R / cnt[:, None]
        R[cnt < MIN_CS_COUNT, :] = np.nan
        return R

    def zscore(Wa: np.ndarray) -> np.ndarray:
        cnt = cnt_of(Wa)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(Wa, axis=1)
            sd = np.nanstd(Wa, axis=1, ddof=1)
            Z = (Wa - mu[:, None]) / sd[:, None]
        Z = np.where(np.isfinite(Z), Z, np.nan)
        Z[cnt < MIN_CS_COUNT, :] = np.nan
        return Z

    def demean(Wa: np.ndarray) -> np.ndarray:
        cnt = cnt_of(Wa)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(Wa, axis=1)
            D = Wa - mu[:, None]
        D = np.where(np.isfinite(D), D, np.nan)
        D[cnt < MIN_CS_COUNT, :] = np.nan
        return D

    def assign(name: str, src_col: str, transform) -> None:
        W = wide_scatter(MAT[:, col_idx[src_col]].astype(np.float64), dc, bc, nD, nB)
        MAT[:, col_idx[name]] = transform(W)[dc, bc]

    for src, name in CS_RANK_SRC:
        assign(name, src, rank_pct)
    for src, name in CS_Z_SRC:
        assign(name, src, zscore)
    for src, name in CS_DEMEAN_SRC:
        assign(name, src, demean)
    for src, name in CS_PASSTHRU:
        MAT[:, col_idx[name]] = passthrough[src]


# --------------------------------------------------------------------------- #
# IC 分析
# --------------------------------------------------------------------------- #
def rowwise_rank_corr(F: np.ndarray, T: np.ndarray) -> np.ndarray:
    """逐行(时间戳) Pearson(秩) = 横截面 Spearman IC. F/T 为按行 rank 过的宽矩阵."""
    mask = np.isfinite(F) & np.isfinite(T)
    n = mask.sum(axis=1).astype(np.float64)
    x = np.where(mask, F, 0.0)
    y = np.where(mask, T, 0.0)
    sx, sy = x.sum(1), y.sum(1)
    sxx, syy, sxy = (x * x).sum(1), (y * y).sum(1), (x * y).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        den = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        ic = np.where((n >= MIN_CS_COUNT) & (den > EPS),
                      cov / np.where(den > EPS, den, 1.0), np.nan)
    return ic


def seg_means_from_rows(rows: np.ndarray, seg_of_date: np.ndarray, n_seg: int) -> np.ndarray:
    finite = np.isfinite(rows)
    cnt = np.bincount(seg_of_date[finite], minlength=n_seg)
    s = np.bincount(seg_of_date[finite], weights=rows[finite], minlength=n_seg)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return np.where(cnt >= MIN_IC_OBS_PER_SEG, m, np.nan)


def pooled_seg_means(feat: np.ndarray, tgt_rank: np.ndarray, seg_row: np.ndarray,
                     n_seg: int) -> np.ndarray:
    """段内 pooled Spearman (用于横截面内常数的日历特征)."""
    fr = np.full(len(feat), np.nan)
    for s in range(n_seg):
        m = seg_row == s
        if m.any():
            fr[m] = rankdata(feat[m], nan_policy="omit")
    mask = np.isfinite(fr) & np.isfinite(tgt_rank)
    ff = np.where(mask, fr, 0.0)
    tt = np.where(mask, tgt_rank, 0.0)
    n = np.bincount(seg_row[mask], minlength=n_seg).astype(np.float64)
    sx = np.bincount(seg_row[mask], weights=ff[mask], minlength=n_seg)
    sy = np.bincount(seg_row[mask], weights=tt[mask], minlength=n_seg)
    sxx = np.bincount(seg_row[mask], weights=(ff * ff)[mask], minlength=n_seg)
    syy = np.bincount(seg_row[mask], weights=(tt * tt)[mask], minlength=n_seg)
    sxy = np.bincount(seg_row[mask], weights=(ff * tt)[mask], minlength=n_seg)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        den = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        r = np.where((n >= 100) & (den > EPS), cov / np.where(den > EPS, den, 1.0), np.nan)
    return r


def summarize(seg_means: np.ndarray) -> dict:
    v = seg_means[np.isfinite(seg_means)]
    res = {"n_seg": int(len(v)), "ic_mean": None, "ic_std": None, "tstat": None, "ic_ir": None}
    if len(v) < MIN_SEG_FOR_STATS or len(v) < 2:
        return res
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1))
    res["ic_mean"] = m
    res["ic_std"] = s
    if np.isfinite(s) and s > EPS:
        res["tstat"] = m / s * np.sqrt(len(v))
        res["ic_ir"] = m / s
    return res


def run_ic_analysis(
    S: np.ndarray,
    feat_cols: list[str],
    col_idx: dict[str, int],
    dc_s: np.ndarray,
    bc_s: np.ndarray,
    nD_s: int,
    nB_s: int,
    n_seg: int,
) -> dict[str, dict]:
    """对每个候选特征计算相对 target_4bar/target_16bar 的分段 Spearman IC.

    dc_s/bc_s 为任意整数编码 (允许来自全量 factorize); 内部压缩成
    0..nD-1 / 0..nB-1 的稠密编码再散射到宽矩阵.
    """
    udates, dc = np.unique(dc_s, return_inverse=True)
    ubases, bc = np.unique(bc_s, return_inverse=True)
    nD, nB = len(udates), len(ubases)
    if nD != nD_s or nB != nB_s:
        log(f"  [WARN] IC dims compacted: dates {nD_s}->{nD}, bases {nB_s}->{nB}")
    seg_of_date = np.minimum((np.arange(nD) * n_seg) // nD, n_seg - 1)
    seg_of_row = seg_of_date[dc]

    # 逐时间戳 rank 过的目标宽矩阵 (横截面 IC 用)
    tgt_rank = {}
    for t in TARGET_COLS:
        W = wide_scatter(S[:, col_idx[t]].astype(np.float64), dc, bc, nD, nB)
        tgt_rank[t] = rankdata(W, axis=1, nan_policy="omit")

    # 段内 pooled rank 的目标 (日历特征 IC 用)
    pooled_t = {}
    for t in TARGET_COLS:
        tv = S[:, col_idx[t]].astype(np.float64)
        tr = np.full(len(tv), np.nan)
        for s in range(n_seg):
            m = seg_of_row == s
            if m.any():
                tr[m] = rankdata(tv[m], nan_policy="omit")
        pooled_t[t] = tr

    results: dict[str, dict] = {}
    for i, name in enumerate(feat_cols):
        x = S[:, col_idx[name]].astype(np.float64)
        entry: dict = {
            "method": "pooled" if name in CALENDAR_FEATURES else "xs",
            "nan_rate": float(np.mean(~np.isfinite(x))),
        }
        if name in CALENDAR_FEATURES:
            for t in TARGET_COLS:
                key = "4" if t == "target_4bar" else "16"
                st = summarize(pooled_seg_means(x, pooled_t[t], seg_of_row, n_seg))
                for k, val in st.items():
                    entry[f"{k}_{key}"] = val
        else:
            W = wide_scatter(x, dc, bc, nD, nB)
            F = rankdata(W, axis=1, nan_policy="omit")
            for t in TARGET_COLS:
                key = "4" if t == "target_4bar" else "16"
                rows = rowwise_rank_corr(F, tgt_rank[t])
                st = summarize(seg_means_from_rows(rows, seg_of_date, n_seg))
                for k, val in st.items():
                    entry[f"{k}_{key}"] = val
        results[name] = entry
        if (i + 1) % 20 == 0:
            log(f"  IC computed for {i + 1}/{len(feat_cols)} features")
    return results


# --------------------------------------------------------------------------- #
# 选择: 过滤 + 聚类 + 回填/截断
# --------------------------------------------------------------------------- #
def _ok(m, t, th_m: float, th_t: float) -> bool:
    return bool(np.isfinite(m) and np.isfinite(t) and abs(m) > th_m and abs(t) > th_t)


def select_features(
    ic: dict[str, dict],
    corr: np.ndarray,
    ranked: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, dict], list[dict]]:
    """返回 (选中特征有序列表, 每特征详细信息, 聚类摘要)."""
    pos = {n: i for i, n in enumerate(ranked)}
    info: dict[str, dict] = {}

    for name in ranked:
        e = ic[name]
        ir4, ir16 = e.get("ic_ir_4"), e.get("ic_ir_16")
        cand = [abs(x) for x in (ir4, ir16) if x is not None and np.isfinite(x)]
        score = max(cand) if cand else -1.0
        best = "16" if (ir16 is not None and np.isfinite(ir16) and
                        (ir4 is None or not np.isfinite(ir4) or abs(ir16) >= abs(ir4))) else "4"
        passed = _ok(e.get("ic_mean_16"), e.get("tstat_16"), args.min_ic, args.t_stat) or \
            _ok(e.get("ic_mean_4"), e.get("tstat_4"), args.min_ic, args.t_stat)
        info[name] = {
            **{k: (float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v)
               for k, v in e.items()},
            "score": float(score),
            "best_horizon": best,
            "passed_filter": bool(passed),
            "cluster": None,
            "cluster_winner": False,
            "selected": False,
            "backfilled": False,
        }

    pool = [n for n in ranked if info[n]["passed_filter"]]
    log(f"filter: {len(pool)}/{len(ranked)} candidates pass "
        f"|IC|>{args.min_ic} & |t|>{args.t_stat}")

    # ---- 层次聚类去冗余 ----
    winners: list[str] = []
    cluster_summary: list[dict] = []
    if len(pool) >= 2:
        pidx = [pos[n] for n in pool]
        sub = corr[np.ix_(pidx, pidx)]
        dist = 1.0 - np.abs(sub)
        dist = np.nan_to_num(dist, nan=1.0)
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, 1.0)
        Z = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(Z, t=1.0 - args.corr_threshold, criterion="distance")
        clusters: dict[int, list[str]] = {}
        for n, lab in zip(pool, labels):
            clusters.setdefault(int(lab), []).append(n)
        for lab, members in clusters.items():
            members_sorted = sorted(members, key=lambda n: info[n]["score"], reverse=True)
            win = members_sorted[0]
            winners.append(win)
            for n in members:
                info[n]["cluster"] = lab
            info[win]["cluster_winner"] = True
            cluster_summary.append({
                "cluster": lab,
                "winner": win,
                "n_members": len(members),
                "pruned": [
                    {"feature": n, "abs_corr_to_winner": float(abs(corr[pos[n], pos[win]]))}
                    for n in members_sorted[1:]
                ],
            })
        winners.sort(key=lambda n: info[n]["score"], reverse=True)
        log(f"clustering: {len(clusters)} clusters from {len(pool)} passers "
            f"(corr threshold {args.corr_threshold})")
    else:
        winners = list(pool)

    selected = list(winners)

    # ---- 回填至 min_selected ----
    if len(selected) < args.min_selected:
        log(f"only {len(selected)} after clustering, backfilling to {args.min_selected}")
        for name in ranked:                       # 优先: 分数高且与已选中相关 <= 阈值
            if len(selected) >= args.min_selected:
                break
            if name in selected:
                continue
            mx = max((abs(corr[pos[name], pos[s]]) for s in selected), default=0.0)
            if np.isfinite(mx) and mx <= args.corr_threshold:
                selected.append(name)
                info[name]["backfilled"] = True
        for name in ranked:                       # 兜底: 无视相关性也要凑满
            if len(selected) >= args.min_selected:
                break
            if name not in selected:
                selected.append(name)
                info[name]["backfilled"] = True
        selected.sort(key=lambda n: info[n]["score"], reverse=True)

    # ---- 截断至 max_selected ----
    if len(selected) > args.max_selected:
        log(f"{len(selected)} selected, trimming to {args.max_selected}")
        selected = selected[: args.max_selected]

    for n in selected:
        info[n]["selected"] = True
    for rank_i, n in enumerate(ranked, start=1):
        info[n]["rank"] = rank_i
    return selected, info, cluster_summary


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
def validate_outputs(expected_cols: list[str], n_rows_expected: int) -> None:
    import pyarrow.feather as pf

    log("===== VALIDATION =====")
    if not OUT_FEATHER.exists() or OUT_FEATHER.stat().st_size == 0:
        raise RuntimeError(f"missing/empty output: {OUT_FEATHER}")
    tbl = pf.read_table(OUT_FEATHER, memory_map=True)
    cols, nrows = list(tbl.schema.names), tbl.num_rows
    del tbl
    log(f"feature_matrix.feather: {nrows:,} rows x {len(cols)} cols "
        f"({OUT_FEATHER.stat().st_size / 1e6:.1f} MB)")
    if nrows != n_rows_expected:
        raise RuntimeError(f"row count mismatch: {nrows} != {n_rows_expected}")
    if cols != expected_cols:
        raise RuntimeError(f"column mismatch:\n  got      {cols}\n  expected {expected_cols}")
    with open(OUT_JSON, encoding="utf-8") as fh:
        rep = json.load(fh)
    log(f"feature_report.json: {rep['n_candidates']} candidates, "
        f"{rep['n_pass_filter']} pass, {rep['n_selected']} selected")
    if not (30 <= rep["n_selected"] <= 120):
        raise RuntimeError(f"implausible selected count: {rep['n_selected']}")
    log("OK: outputs verified")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-rows", type=int, default=500_000,
                   help="IC 分析抽样行数 (按时间戳 stride 采样)")
    p.add_argument("--n-segments", type=int, default=20)
    p.add_argument("--min-ic", type=float, default=0.008)
    p.add_argument("--t-stat", type=float, default=1.5)
    p.add_argument("--corr-threshold", type=float, default=0.7)
    p.add_argument("--min-selected", type=int, default=50)
    p.add_argument("--max-selected", type=int, default=80)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading {IN_FEATHER.name} ...")
    df = pd.read_feather(IN_FEATHER)
    n_rows = len(df)
    df = df.sort_values(["date", "base"], ignore_index=True)
    log(f"loaded {n_rows:,} rows, {df['base'].nunique()} coins, "
        f"{df['date'].min()} -> {df['date'].max()}")

    passthrough = {
        "_cs_volume": df["_cs_volume"].to_numpy(np.float32),
        "_cs_funding": df["_cs_funding"].to_numpy(np.float32),
    }

    # ---- 阶段 1: 逐币特征 -> 预分配矩阵 ----
    MAT: np.ndarray | None = None
    all_cols: list[str] = []
    col_idx: dict[str, int] = {}
    n_ts_cols = 0
    n_coins = df["base"].nunique()
    for i, (base, g) in enumerate(df.groupby("base", sort=True)):
        fg = per_coin_features(g)
        if MAT is None:
            n_ts_cols = fg.shape[1]
            all_cols = list(fg.columns) + CS_FEATURE_NAMES
            col_idx = {n: j for j, n in enumerate(all_cols)}
            MAT = np.empty((n_rows, len(all_cols)), np.float32)
        pos = g.index.to_numpy()
        MAT[pos, :n_ts_cols] = fg.to_numpy(np.float32)
        del fg
        if (i + 1) % 9 == 0 or i == n_coins - 1:
            log(f"per-coin features: {i + 1}/{n_coins} coins done")
    feature_cols = [c for c in all_cols if c not in TARGET_COLS]
    log(f"feature matrix: {n_rows:,} x {len(all_cols)} "
        f"({len(feature_cols)} candidates + 2 targets)")

    # ---- 阶段 2: 横截面特征 ----
    dc, duniq = pd.factorize(df["date"])
    bc, buniq = pd.factorize(df["base"])
    add_cross_sectional(MAT, col_idx, dc, bc, len(duniq), len(buniq), passthrough)
    gc.collect()
    log("cross-sectional features done")
    bad = [all_cols[j] for j in range(MAT.shape[1]) if not np.isfinite(MAT[:, j]).any()]
    if bad:
        raise RuntimeError(f"all-NaN feature columns: {bad}")

    # ---- 阶段 3: 抽样 + IC ----
    stride = max(1, int(round(n_rows / args.sample_rows)))
    keep = np.zeros(len(duniq), dtype=bool)
    keep[::stride] = True
    sample_idx = np.flatnonzero(keep[dc])
    S = MAT[sample_idx]
    dc_s, bc_s = dc[sample_idx], bc[sample_idx]
    nD_s, nB_s = int(keep.sum()), len(buniq)
    log(f"IC sample: {len(sample_idx):,} rows / {nD_s:,} timestamps (stride={stride})")
    ic = run_ic_analysis(S, feature_cols, col_idx, dc_s, bc_s, nD_s, nB_s, args.n_segments)

    # ---- 阶段 4: 打分排序 + 相关矩阵 ----
    def score_of(name: str) -> float:
        e = ic[name]
        cand = [abs(e[k]) for k in ("ic_ir_4", "ic_ir_16")
                if e.get(k) is not None and np.isfinite(e[k])]
        return max(cand) if cand else -1.0

    ranked = sorted(feature_cols, key=score_of, reverse=True)
    ranked = [n for n in ranked if score_of(n) > 0]
    log(f"scorable candidates: {len(ranked)}/{len(feature_cols)}")
    pidx = [col_idx[n] for n in ranked]
    log("computing Spearman correlation matrix on sample ...")
    corr = (pd.DataFrame(S[:, pidx], columns=ranked)
            .rank()
            .corr(min_periods=1000)
            .to_numpy())
    del pidx
    gc.collect()

    # ---- 阶段 5: 选择 ----
    selected, info, cluster_summary = select_features(ic, corr, ranked, args)
    log(f"selected {len(selected)} features")

    # ---- 阶段 6: 输出 ----
    out_cols = ["date", "base", *selected, *TARGET_COLS]
    data: dict = {"date": df["date"], "base": df["base"]}
    for n in selected:
        data[n] = MAT[:, col_idx[n]]
    for t in TARGET_COLS:
        data[t] = MAT[:, col_idx[t]]
    out = pd.DataFrame(data, columns=out_cols)
    del data
    out.to_feather(OUT_FEATHER)
    log(f"saved {OUT_FEATHER} ({OUT_FEATHER.stat().st_size / 1e6:.1f} MB)")

    nan_rates = {n: float(np.mean(~np.isfinite(MAT[:, col_idx[n]]))) for n in selected}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": round(time.time() - T0, 1),
        "input": {
            "feather": str(IN_FEATHER), "rows": int(n_rows),
            "coins": int(n_coins),
            "date_min": str(df["date"].min()), "date_max": str(df["date"].max()),
        },
        "sample": {
            "rows": int(len(sample_idx)), "timestamps": nD_s, "stride": stride,
            "n_segments": args.n_segments,
            "segment_method": "mean of per-timestamp cross-sectional Spearman IC "
                              "(calendar features: pooled per-segment Spearman)",
        },
        "targets": {
            "target_4bar": "future 4-bar simple return (only forward-looking column)",
            "target_16bar": "future 16-bar simple return",
        },
        "thresholds": {
            "min_abs_ic_mean": args.min_ic, "min_abs_tstat": args.t_stat,
            "corr_threshold": args.corr_threshold,
            "min_selected": args.min_selected, "max_selected": args.max_selected,
        },
        "n_candidates": len(feature_cols),
        "n_pass_filter": int(sum(1 for n in ranked if info[n]["passed_filter"])),
        "n_clusters": len(cluster_summary),
        "n_selected": len(selected),
        "selected_features": selected,
        "selected_nan_rate": nan_rates,
        "clusters": cluster_summary,
        "features": {n: info[n] for n in sorted(feature_cols, key=lambda x: info.get(x, {}).get("rank", 10**9))},
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved {OUT_JSON}")

    # ---- 摘要 ----
    log("===== TOP SELECTED (by |IC IR|) =====")
    for n in selected[:20]:
        e = ic[n]
        log(f"  {n:<22s} method={info[n]['method']:<6s} "
            f"IC16={e.get('ic_mean_16', float('nan')):+.4f} "
            f"t16={e.get('tstat_16', float('nan')):+.2f} "
            f"IC4={e.get('ic_mean_4', float('nan')):+.4f} "
            f"t4={e.get('tstat_4', float('nan')):+.2f} "
            f"backfill={info[n]['backfilled']}")

    validate_outputs(out_cols, n_rows)
    log(f"done in {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""模型层: Purged Walk-Forward CV + 多模型集成 + Meta-Labeling (Lopez de Prado)
==========================================================================

输入 : user_data/ml/feature_matrix.feather   (特征层输出; 若缺失且 feature_factory.py
                                               存在则先调用之, 否则用内置 fallback 特征)
输出 : user_data/ml/predictions.feather      每根 bar 预测 (date, base, pred_ret,
                                               pred_direction, pred_confidence, signal)
       user_data/ml/model_performance.json   每轮 WFCV 指标 + 汇总
       user_data/ml/feature_importance.json  各模型特征重要度

方法:
  1) Purged Walk-Forward CV
       训练 12 个月 → 测试 3 个月, 滚动步长 3 个月
       Purging : 训练集末尾 label_period 根 bar 的样本删除(其标签窗口与测试期重叠)
       Embargo : 测试窗口前 10% 时间为隔离带, 只预测不参与评估
  2) 集成: LightGBM + XGBoost + Ridge 三个回归模型预测前瞻收益, 预测值简单平均
  3) Meta-labeling 双层模型:
       L1 方向模型   LGBMClassifier, 预测未来收益方向 (>0 / <0)
       L2 置信度模型 LGBMClassifier, 输入 = 原始特征 + L1 (概率, 方向),
                     预测 "L1 方向是否正确"; L2 的训练标签用训练窗内
                     expanding OOF 的 L1 输出构造, 防止 L1 记忆泄漏进 L2
       最终信号 = L1方向 × (L2置信度 > 0.6)
  4) 评估: AUC / 高置信信号精确率召回率 / 预测值十分位实际收益 / Spearman IC

用法: python scripts/ml/model_engine.py [--start 2024-01] [--max-rows 700000]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = ROOT / "user_data" / "ml"
DATA_DIR = ROOT / "user_data" / "data" / "binance" / "futures"
FEATURE_MATRIX = ML_DIR / "feature_matrix.feather"
FEATURE_FACTORY = ROOT / "scripts" / "ml" / "feature_factory.py"

# ---- WFCV 配置 ----
TRAIN_MONTHS = 12          # 训练窗口
TEST_MONTHS = 3            # 测试窗口
STEP_MONTHS = 3            # 滚动步长
EMBARGO_PCT = 0.10         # 测试窗口前 10% 时间隔离带
CONF_THRESHOLD = 0.60      # meta-labeling 置信度阈值
DEFAULT_LABEL_BARS = 16    # 兜底标签期(15m × 16 = 4h), 实际从标签列名/数据推断
MAX_TRAIN_ROWS = 700_000   # 单折训练行数上限(随机子采样, 控制运行时间)
OOF_MAX_ROWS = 450_000     # L2 OOF 内层单次拟合行数上限
MIN_TRAIN_ROWS = 30_000
MIN_EVAL_ROWS = 5_000
SEED = 42

DATE_COL_CANDIDATES = ("date", "datetime", "timestamp")
BASE_COL_CANDIDATES = ("base", "coin", "symbol", "asset", "pair")
META_EXCLUDE = {
    "date", "datetime", "timestamp", "base", "coin", "symbol", "asset", "pair",
    "open", "high", "low", "close", "volume",
}
LABEL_PAT = re.compile(r"(fwd|forward|future|fut)[_\-]?(ret|return|logret|log_ret)|^target_\d+bar$", re.I)
BINARY_LABEL_PAT = re.compile(r"^(label|target|y|bin|class)[_0-9]*$", re.I)


def log(msg: str) -> None:
    print(msg, flush=True)


# ===========================================================================
# 数据加载: 优先特征层产物, 其次调用特征层脚本, 最后内置 fallback
# ===========================================================================

def load_feature_matrix(max_rows: int | None = None) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, int, str]:
    """返回 (meta[date,base], y_ret, X, label_bars, source)"""
    if not FEATURE_MATRIX.exists() and FEATURE_FACTORY.exists():
        log("特征矩阵不存在, 先运行特征层脚本 (可能需要 5-10 分钟)...")
        subprocess.run([sys.executable, str(FEATURE_FACTORY)], check=False)

    if FEATURE_MATRIX.exists():
        df = pd.read_feather(FEATURE_MATRIX)
        source = f"feature_matrix ({FEATURE_MATRIX.name})"
    else:
        log("[FALLBACK] 未找到特征层产物, 使用内置 fallback 特征 (项目既有 15m 特征集)...")
        df = build_fallback_matrix()
        source = "fallback_features (内置, 与 ml_study.py 同源)"

    if "date" not in [c.lower() for c in df.columns] and df.index.name in DATE_COL_CANDIDATES:
        df = df.reset_index()

    date_col = next((c for c in df.columns if c.lower() in DATE_COL_CANDIDATES), None)
    base_col = next((c for c in df.columns if c.lower() in BASE_COL_CANDIDATES), None)
    if date_col is None:
        raise SystemExit(f"特征矩阵缺少日期列, 现有列: {list(df.columns)[:30]}")
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # 标签列: 前瞻收益(连续). 从列名提取 bar 数, 否则用数据推断的 bar 周期 × 默认
    label_col = next((c for c in df.columns if LABEL_PAT.search(c)), None)
    if label_col is None:
        raise SystemExit(f"特征矩阵缺少前瞻收益标签列 (fwd_ret/forward_return...), 现有列: {list(df.columns)[:30]}")
    m = re.search(r"(\d+)", label_col)
    bar_interval = infer_bar_interval(df["date"], df.get(base_col) if base_col else None)
    label_bars = int(m.group(1)) if m else max(1, int(pd.Timedelta(hours=4) / bar_interval))

    feature_cols = [
        c for c in df.columns
        if c not in META_EXCLUDE and c != label_col and c != "date"
        and not LABEL_PAT.search(c) and not BINARY_LABEL_PAT.match(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(feature_cols) < 5:
        raise SystemExit(f"特征列过少({len(feature_cols)}), 列样本: {list(df.columns)[:30]}")

    y = pd.to_numeric(df[label_col], errors="coerce").astype("float32").to_numpy()
    valid = np.isfinite(y)
    y = y[valid]
    meta = pd.DataFrame({
        "date": df.loc[valid, "date"].to_numpy(),
        "base": (df.loc[valid, base_col].astype(str).to_numpy() if base_col
                 else np.array(["ALL"] * int(valid.sum()))),
    })
    X = df.loc[valid, feature_cols].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)
    valid2 = X.notna().mean(axis=1).to_numpy() >= 0.8   # 允许少量缺失(中位数填补)
    meta, y, X = meta.loc[valid2].reset_index(drop=True), y[valid2], X.loc[valid2].reset_index(drop=True)

    if max_rows and len(X) > max_rows:  # 全局演示上限(默认不用)
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(X), max_rows, replace=False))
        meta, y, X = meta.iloc[idx].reset_index(drop=True), y[idx], X.iloc[idx].reset_index(drop=True)

    log(f"特征矩阵: {len(X):,} 行 × {X.shape[1]} 特征 | {meta['date'].min()} → {meta['date'].max()} | "
        f"标签列={label_col}, label_bars={label_bars} ({label_bars * bar_interval}) | 来源: {source}")
    return meta, y, X, label_bars, source


def infer_bar_interval(dates: pd.Series, bases: pd.Series | None) -> pd.Timedelta:
    if bases is not None:
        b = bases.iloc[0]
        d = dates[bases == b]
    else:
        d = dates
    diff = d.diff().dropna()
    med = diff.median() if len(diff) else pd.Timedelta(minutes=15)
    return med if med > pd.Timedelta(0) else pd.Timedelta(minutes=15)


def build_fallback_matrix() -> pd.DataFrame:
    """与 scripts/ml_study.py 同源的 15m 特征集 + 4h 前瞻收益(仅在特征层缺失时兜底)"""
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    bases = [p.split("/")[0] for p in cfg["exchange"]["pair_whitelist"]]
    data = {}
    for b in bases:
        f = DATA_DIR / f"{b}_USDT_USDT-15m-futures.feather"
        if f.exists():
            data[b] = pd.read_feather(f)
    all_closes = pd.DataFrame({b: d.set_index("date")["close"] for b, d in data.items()})
    market_mean = all_closes.mean(axis=1)

    H = 16  # 4h
    feats, outs = [], []
    for b, df in data.items():
        c, o, h, l, v = (df[k].to_numpy() for k in ("close", "open", "high", "low", "volume"))
        n = len(c)
        cs = pd.Series(c)
        feat = pd.DataFrame(index=df.index)
        for lb in (4, 8, 16, 48, 96, 288):
            feat[f"ret_{lb}"] = cs.pct_change(lb)
        feat["dist_ema20"] = cs / cs.ewm(span=20).mean().to_numpy() - 1
        feat["dist_ema50"] = cs / cs.ewm(span=50).mean().to_numpy() - 1
        feat["dist_ema200"] = cs / cs.ewm(span=200).mean().to_numpy() - 1
        mid, sd = cs.rolling(20).mean(), cs.rolling(20).std()
        feat["bb_pos"] = (cs - mid) / (2 * sd + 1e-12)
        feat["bb_width"] = (4 * sd) / (mid + 1e-12)
        feat["bb_width_pct"] = feat["bb_width"] / feat["bb_width"].rolling(96).mean()
        for p in (7, 14, 28):
            dlt = cs.diff()
            up = dlt.where(dlt > 0, 0).ewm(alpha=1 / p).mean()
            dn = (-dlt.where(dlt < 0, 0)).ewm(alpha=1 / p).mean()
            feat[f"rsi_{p}"] = 100 - 100 / (1 + up / (dn + 1e-12))
        feat["body_ratio"] = np.abs(c - o) / (h - l + 1e-12)
        feat["upper_wick"] = (h - np.maximum(c, o)) / (h - l + 1e-12)
        feat["lower_wick"] = (np.minimum(c, o) - l) / (h - l + 1e-12)
        vs = pd.Series(v)
        feat["vol_ratio_4"] = vs / (vs.rolling(4).mean() + 1e-12)
        feat["vol_ratio_96"] = vs / (vs.rolling(96).mean() + 1e-12)
        feat["vol_trend"] = vs.rolling(16).mean() / (vs.rolling(96).mean() + 1e-12)
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        feat["atr_14"] = pd.Series(tr).rolling(14).mean() / c
        feat["atr_ratio"] = pd.Series(tr).rolling(14).mean() / (pd.Series(tr).rolling(96).mean() + 1e-12)
        feat["realized_vol_16"] = cs.pct_change().abs().rolling(16).std()
        ma = market_mean.reindex(df["date"]).ffill().to_numpy()
        mr = np.full(n, np.nan); mr[96:] = ma[96:] / ma[:-96] - 1
        cr = np.full(n, np.nan); cr[96:] = c[96:] / c[:-96] - 1
        feat["relative_strength_24h"] = cr - mr
        feat["htf_trend"] = (cs.ewm(span=200).mean() > cs.ewm(span=800).mean()).astype(int)
        feat["hour"] = df["date"].dt.hour.to_numpy()
        feat["dow"] = df["date"].dt.dayofweek.to_numpy()
        ff = DATA_DIR / f"{b}_USDT_USDT-1h-funding_rate.feather"
        if ff.exists():
            fund = pd.read_feather(ff).set_index("date")["open"].reindex(df["date"]).ffill()
            feat["funding"] = fund.to_numpy()
            feat["funding_ma"] = fund.rolling(12).mean().to_numpy()
        else:
            feat["funding"] = 0.0
            feat["funding_ma"] = 0.0

        fwd = np.full(n, np.nan)
        fwd[:-H] = c[H:] / c[:-H] - 1
        feat["fwd_ret_16"] = fwd
        feat["base"] = b
        feat["date"] = df["date"].to_numpy()
        feats.append(feat)
    return pd.concat(feats, ignore_index=True)


# ===========================================================================
# 模型定义
# ===========================================================================

def make_lgbm_reg() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.01, num_leaves=63,
        subsample=0.7, subsample_freq=1, random_state=SEED, n_jobs=-1,
        verbose=-1, force_col_wise=True,
    )


def make_xgb_reg() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.01, subsample=0.7,
        tree_method="hist", n_jobs=-1, verbosity=0, random_state=SEED,
    )


def make_ridge() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def make_lgbm_clf() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.03, num_leaves=63,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=100, random_state=SEED, n_jobs=-1,
        verbose=-1, force_col_wise=True,
    )


class EnsembleRegressor:
    """LightGBM + XGBoost + Ridge 简单平均集成(回归: 前瞻收益)"""

    def __init__(self):
        self.m_lgb = make_lgbm_reg()
        self.m_xgb = make_xgb_reg()
        self.m_ridge = make_ridge()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleRegressor":
        self.m_lgb.fit(X, y)
        self.m_xgb.fit(X, y)
        self.m_ridge.fit(X, y)
        return self

    def predict_each(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "lgbm": self.m_lgb.predict(X),
            "xgbm": self.m_xgb.predict(X),
            "ridge": self.m_ridge.predict(X),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_each(X)
        return (p["lgbm"] + p["xgbm"] + p["ridge"]) / 3.0


class MetaLabeler:
    """L1 方向模型 + L2 置信度模型 (双层 meta-labeling)

    L2 训练标签基于训练窗内 expanding OOF 的 L1 输出(带 purge), 避免同一行
    被 L1 记忆后 L2 只学到"L1 训练精度≈100%"的假象.
    """

    def __init__(self, horizon_td: pd.Timedelta):
        self.horizon_td = horizon_td
        self.l1 = make_lgbm_clf()
        self.l2 = make_lgbm_clf()
        self.l1_acc_oof_ = np.nan

    def fit(self, X: np.ndarray, dates: pd.Series, y_ret: np.ndarray) -> "MetaLabeler":
        y_dir = (y_ret > 0).astype(int)
        n = len(y_dir)
        # --- 内层 expanding OOF: 时间等分 5 块, 用前 k 块拟合预测第 k 块 (k=3,4,5) ---
        block = pd.qcut(dates, 5, labels=False).to_numpy()
        oof_p = np.full(n, np.nan)
        rng = np.random.default_rng(SEED)
        for k in (2, 3, 4):
            fit_mask = block < k
            pred_mask = block == k
            if not fit_mask.any() or not pred_mask.any():
                continue
            # 内层 purge: 去掉与第 k 块边界重叠标签的训练样本
            boundary = dates[pred_mask].min() - self.horizon_td
            fit_mask &= (dates <= boundary).to_numpy()
            if fit_mask.sum() < 20_000 or pred_mask.sum() < 5_000:
                continue
            fit_idx = np.where(fit_mask)[0]
            if len(fit_idx) > OOF_MAX_ROWS:
                fit_idx = np.sort(rng.choice(fit_idx, OOF_MAX_ROWS, replace=False))
            m = make_lgbm_clf()
            m.fit(X[fit_idx], y_dir[fit_idx])
            oof_p[pred_mask] = m.predict_proba(X[np.where(pred_mask)[0]])[:, 1]

        # --- 用 OOF 结果训练 L2 ---
        mask = np.isfinite(oof_p)
        if mask.sum() < 30_000:  # OOF 数据不足时退化为全量训练(记录警告)
            log("    [warn] OOF 样本不足, L2 退化为全量拟合(可能高估置信度)")
            oof_p = self._full_fit_predict(X, y_dir)
            mask = np.isfinite(oof_p)
        cls1 = (oof_p[mask] > 0.5)
        correct = (cls1 == (y_dir[mask] == 1))
        self.l1_acc_oof_ = float(correct.mean())
        X2 = np.column_stack([X[mask], oof_p[mask], cls1.astype(np.float32)])
        self.l2.fit(X2, correct.astype(int))

        # --- 最终 L1: 全训练窗 ---
        self.l1.fit(X, y_dir)
        return self

    def _full_fit_predict(self, X: np.ndarray, y_dir: np.ndarray) -> np.ndarray:
        m = make_lgbm_clf()
        m.fit(X, y_dir)
        return m.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (direction ±1, confidence, l1_proba)"""
        p1 = self.l1.predict_proba(X)[:, 1]
        cls1 = (p1 > 0.5).astype(np.float32)
        X2 = np.column_stack([X, p1, cls1])
        conf = self.l2.predict_proba(X2)[:, 1]
        direction = np.where(cls1 > 0, 1, -1).astype(np.int8)
        return direction, conf.astype(np.float32), p1.astype(np.float32)


# ===========================================================================
# 评估
# ===========================================================================

def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 100 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def evaluate_fold(preds: dict[str, np.ndarray], direction: np.ndarray, conf: np.ndarray,
                  p1: np.ndarray, y_ret: np.ndarray) -> dict:
    y_dir = (y_ret > 0).astype(int)
    out: dict = {"auc_direction": safe_auc(y_dir, p1)}

    # --- IC: 各模型 + 集成 ---
    ics = {f"ic_{k}": safe_spearman(v, y_ret) for k, v in preds.items()}
    out.update(ics)
    out["auc_ensemble_ret"] = safe_auc(y_dir, preds["ensemble"])

    # --- meta-labeling: 高置信信号精确率/召回率 ---
    fired = conf > CONF_THRESHOLD
    correct = (direction * np.sign(y_ret)) > 0
    out["coverage"] = float(fired.mean())
    out["precision_high_conf"] = float(correct[fired].mean()) if fired.any() else float("nan")
    out["recall_high_conf"] = float(fired[correct].mean()) if correct.any() else float("nan")
    out["accuracy_l1_all"] = float(correct.mean())            # 不加过滤时 L1 命中率(基线)
    out["mean_ret_fired"] = float((direction * y_ret)[fired].mean()) if fired.any() else float("nan")
    out["mean_ret_long"] = float(y_ret[fired & (direction > 0)].mean()) if (fired & (direction > 0)).any() else float("nan")
    out["mean_ret_short"] = float((-y_ret)[fired & (direction < 0)].mean()) if (fired & (direction < 0)).any() else float("nan")

    # --- 十分位: 集成预测值分组后的平均实际收益 ---
    ranks = rankdata(preds["ensemble"])
    dec = np.minimum((ranks / (len(ranks) / 10 + 1)).astype(int), 9)  # 0..9, 值越大预测越高
    decile_mean = {f"d{i+1}": float(y_ret[dec == i].mean()) for i in range(10)}
    out["decile_mean_ret"] = decile_mean
    out["decile_spread"] = decile_mean["d10"] - decile_mean["d1"]
    out["decile_monotonicity"] = safe_spearman(np.arange(10), np.array([decile_mean[f"d{i+1}"] for i in range(10)]))
    return out


def collect_importance(models: dict, feature_names: list[str]) -> dict[str, dict[str, float]]:
    """各模型本折重要度(归一化到和为1)"""
    out = {}

    def norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        s = v.sum()
        return v / s if s > 0 else v

    out["lgbm_regressor"] = dict(zip(feature_names, norm(models["ens"].m_lgb.feature_importances_)))
    out["xgboost_regressor"] = dict(zip(feature_names, norm(models["ens"].m_xgb.feature_importances_)))
    out["direction_model_l1"] = dict(zip(feature_names, norm(models["meta"].l1.feature_importances_)))
    names_l2 = feature_names + ["l1_proba", "l1_class"]
    out["confidence_model_l2"] = dict(zip(names_l2, norm(models["meta"].l2.feature_importances_)))
    ridge_coef = np.abs(models["ens"].m_ridge.named_steps["ridge"].coef_)
    out["ridge_abs_coef"] = dict(zip(feature_names, norm(ridge_coef)))
    return out


# ===========================================================================
# 主流程: Purged Walk-Forward CV
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None, help="首个测试窗起始月, 如 2024-01 (默认自动)")
    ap.add_argument("--max-rows", type=int, default=MAX_TRAIN_ROWS, help="单折训练行数上限")
    args = ap.parse_args()

    ML_DIR.mkdir(parents=True, exist_ok=True)
    meta, y, X, label_bars, source = load_feature_matrix()
    feature_names = list(X.columns)
    horizon_td = infer_bar_interval(meta["date"], meta["base"]) * label_bars

    # --- 测试窗序列: 12 个月训练数据可用后的第一个月初开始, 每 3 个月滚动 ---
    dmin, dmax = meta["date"].min(), meta["date"].max()
    auto_start = (dmin + pd.DateOffset(months=TRAIN_MONTHS))
    start = pd.Timestamp(args.start, tz="UTC") if args.start else auto_start.ceil("D")
    test_starts = pd.date_range(start=start, end=dmax, freq=f"{STEP_MONTHS}MS", tz="UTC")
    log(f"\n=== Purged Walk-Forward: 训练{TRAIN_MONTHS}个月 → 测试{TEST_MONTHS}个月, 步长{STEP_MONTHS}个月 | "
        f"purge={label_bars} bars ({horizon_td}) | embargo={EMBARGO_PCT:.0%} 测试窗时长 ===")
    log(f"测试窗: {len(test_starts)} 轮  {test_starts[0]} → {test_starts[-1]}\n")

    fold_records: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    imp_sum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rng_master = np.random.default_rng(SEED)

    for i, ts in enumerate(test_starts):
        t0 = time.time()
        te_end = ts + pd.DateOffset(months=TEST_MONTHS)
        tr_start = ts - pd.DateOffset(months=TRAIN_MONTHS)
        purge_line = ts - horizon_td                       # Purging 边界
        tr_mask = (meta["date"] >= tr_start) & (meta["date"] < purge_line)
        te_mask = (meta["date"] >= ts) & (meta["date"] < te_end)
        embargo_td = pd.Timedelta(seconds=EMBARGO_PCT * (te_end - ts).total_seconds())
        ev_mask = te_mask & (meta["date"] >= ts + embargo_td)
        n_tr, n_te, n_ev = int(tr_mask.sum()), int(te_mask.sum()), int(ev_mask.sum())
        if n_tr < MIN_TRAIN_ROWS or n_ev < MIN_EVAL_ROWS:
            log(f"[fold {i+1:02d}/{len(test_starts)}] {ts:%Y-%m-%d} → {te_end:%Y-%m-%d}  "
                f"跳过 (train={n_tr:,} eval={n_ev:,})")
            continue

        # 训练子采样(均匀随机, 保分布控时长)
        tr_idx = np.where(tr_mask.to_numpy())[0]
        if len(tr_idx) > args.max_rows:
            tr_idx = np.sort(rng_master.choice(tr_idx, args.max_rows, replace=False))
        X_tr = X.iloc[tr_idx].to_numpy().copy()  # copy: feather 来源数组只读
        y_tr = y[tr_idx]

        # 中位数填补(训练集统计量, 防泄漏)
        med = np.nanmedian(X_tr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        nan_tr = np.isnan(X_tr)
        if nan_tr.any():
            X_tr[nan_tr] = np.take(med, np.where(nan_tr)[1])
        X_te = X[te_mask].to_numpy().copy()
        nan_te = np.isnan(X_te)
        if nan_te.any():
            X_te[nan_te] = np.take(med, np.where(nan_te)[1])

        # --- 集成回归 ---
        ens = EnsembleRegressor().fit(X_tr, y_tr)
        p_each = ens.predict_each(X_te)
        p_ens_te = (p_each["lgbm"] + p_each["xgbm"] + p_each["ridge"]) / 3.0

        # --- Meta-labeling 双层模型 ---
        meta_lb = MetaLabeler(horizon_td).fit(X_tr, meta["date"].iloc[tr_idx].reset_index(drop=True), y_tr)
        direction_te, conf_te, p1_te = meta_lb.predict(X_te)

        # --- 评估(仅隔离带之后) ---
        ev = ev_mask[te_mask].to_numpy()
        preds_ev = {
            "ensemble": p_ens_te[ev],
            "lgbm": p_each["lgbm"][ev],
            "xgbm": p_each["xgbm"][ev],
            "ridge": p_each["ridge"][ev],
        }
        metrics = evaluate_fold(preds_ev, direction_te[ev], conf_te[ev], p1_te[ev], y[te_mask.to_numpy()][ev])
        metrics.update({
            "fold": i + 1,
            "test_start": str(ts.date()), "test_end": str(te_end.date()),
            "embargo_days": round(embargo_td.total_seconds() / 86400, 1),
            "n_train": n_tr, "n_test": n_te, "n_eval": int(ev.sum()),
            "l1_oof_accuracy": round(meta_lb.l1_acc_oof_, 4),
            "elapsed_sec": round(time.time() - t0, 1),
        })
        fold_records.append(metrics)

        # --- 预测输出(整个测试窗, 含隔离带) ---
        pred_frames.append(pd.DataFrame({
            "date": meta.loc[te_mask, "date"].to_numpy(),
            "base": meta.loc[te_mask, "base"].to_numpy(),
            "pred_ret": p_ens_te.astype("float32"),
            "pred_direction": direction_te,
            "pred_confidence": conf_te,
            "signal": (direction_te * (conf_te > CONF_THRESHOLD).astype(np.int8)),
        }))

        # --- 特征重要度 ---
        for model_name, imp in collect_importance({"ens": ens, "meta": meta_lb}, feature_names).items():
            for f, v in imp.items():
                imp_sum[model_name][f].append(float(v))

        log(f"[fold {i+1:02d}/{len(test_starts)}] {ts:%Y-%m} → {te_end:%Y-%m} | "
            f"AUC={metrics['auc_direction']:.4f} IC_ens={metrics['ic_ensemble']:+.4f} "
            f"(lgb {metrics['ic_lgbm']:+.3f} xgb {metrics['ic_xgbm']:+.3f} ridge {metrics['ic_ridge']:+.3f}) | "
            f"P_hc={metrics['precision_high_conf']:.3f} vs acc={metrics['accuracy_l1_all']:.3f} | "
            f"cov={metrics['coverage']:.1%} | spread={metrics['decile_spread']:+.5f} | "
            f"{metrics['elapsed_sec']:.0f}s")

    if not fold_records:
        raise SystemExit("没有可用的 WFCV 窗口, 检查数据时间跨度与 --start")

    # ======================= 输出 1: predictions.feather =======================
    preds_df = pd.concat(pred_frames, ignore_index=True).sort_values(["date", "base"]).reset_index(drop=True)
    preds_df.to_feather(ML_DIR / "predictions.feather")
    log(f"\n预测结果 → {ML_DIR / 'predictions.feather'}  ({len(preds_df):,} 行, "
        f"{preds_df['date'].min()} → {preds_df['date'].max()}, 信号≠0 占比 "
        f"{(preds_df['signal'] != 0).mean():.1%})")

    # ======================= 输出 2: model_performance.json ====================
    def col(key: str) -> np.ndarray:
        return np.array([r[key] for r in fold_records], dtype=float)

    auc = col("auc_direction")
    ics = {k: col(f"ic_{k}") for k in ("ensemble", "lgbm", "xgbm", "ridge")}
    prec = col("precision_high_conf")
    acc = col("accuracy_l1_all")
    valid_pr = np.isfinite(prec) & np.isfinite(acc)
    spread = col("decile_spread")

    summary = {
        "n_folds": len(fold_records),
        "eval_rows_total": int(col("n_eval").sum()),
        "auc_direction": {
            "mean": float(np.nanmean(auc)), "std": float(np.nanstd(auc)),
            "min": float(np.nanmin(auc)), "max": float(np.nanmax(auc)),
            "target_0.58": bool(np.nanmean(auc) > 0.58),
        },
        "ic_spearman": {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))} for k, v in ics.items()},
        "ensemble_vs_single": {
            "ensemble_beats_lgbm_folds": int(np.nansum(ics["ensemble"] > ics["lgbm"])),
            "ensemble_beats_xgbm_folds": int(np.nansum(ics["ensemble"] > ics["xgbm"])),
            "ensemble_beats_ridge_folds": int(np.nansum(ics["ensemble"] > ics["ridge"])),
            "note": f"共 {len(fold_records)} 折; 集成平均IC {np.nanmean(ics['ensemble']):+.4f} "
                    f"vs 最佳单模型 {max(np.nanmean(v) for k, v in ics.items() if k != 'ensemble'):+.4f}",
        },
        "meta_labeling": {
            "mean_l1_accuracy_baseline": float(np.nanmean(acc)),
            "mean_high_conf_precision": float(np.nanmean(prec)),
            "precision_lift": float(np.nanmean(prec[valid_pr]) - np.nanmean(acc[valid_pr])),
            "mean_high_conf_recall": float(np.nanmean(col("recall_high_conf"))),
            "mean_signal_coverage": float(np.nanmean(col("coverage"))),
            "mean_fired_signal_return": float(np.nanmean(col("mean_ret_fired"))),
            "confidence_threshold": CONF_THRESHOLD,
        },
        "decile": {
            "mean_top_decile_ret": float(np.nanmean([r["decile_mean_ret"]["d10"] for r in fold_records])),
            "mean_bottom_decile_ret": float(np.nanmean([r["decile_mean_ret"]["d1"] for r in fold_records])),
            "mean_spread_top_minus_bottom": float(np.nanmean(spread)),
            "mean_monotonicity": float(np.nanmean(col("decile_monotonicity"))),
        },
    }

    perf = {
        "config": {
            "train_months": TRAIN_MONTHS, "test_months": TEST_MONTHS, "step_months": STEP_MONTHS,
            "embargo_pct": EMBARGO_PCT, "label_bars": label_bars, "horizon": str(horizon_td),
            "confidence_threshold": CONF_THRESHOLD, "feature_source": source,
            "models": {
                "lgbm_reg": "n_estimators=300, max_depth=8, lr=0.01, num_leaves=63, subsample=0.7",
                "xgb_reg": "n_estimators=300, max_depth=8, lr=0.01, subsample=0.7",
                "ridge": "alpha=1.0 (StandardScaler pipeline)",
                "ensemble": "三者预测简单平均",
                "meta_l1": "LGBMClassifier 方向模型 (400 trees, lr=0.03)",
                "meta_l2": "LGBMClassifier 置信度模型 (特征+L1概率+L1方向, OOF标签)",
            },
        },
        "folds": fold_records,
        "summary": summary,
    }

    def round_floats(o, nd=6):
        if isinstance(o, (float, np.floating)):
            v = float(o)
            return round(v, nd) if np.isfinite(v) else None
        if isinstance(o, dict):
            return {k: round_floats(v, nd) for k, v in o.items()}
        if isinstance(o, list):
            return [round_floats(v, nd) for v in o]
        return o

    (ML_DIR / "model_performance.json").write_text(
        json.dumps(round_floats(perf), indent=2, ensure_ascii=False), encoding="utf-8")

    # ======================= 输出 3: feature_importance.json ===================
    imp_out = {}
    for model_name, feats in imp_sum.items():
        avg = {f: float(np.mean(v)) for f, v in feats.items()}
        imp_out[model_name] = dict(sorted(avg.items(), key=lambda kv: -kv[1]))
    tree_models = ["lgbm_regressor", "xgboost_regressor", "direction_model_l1", "confidence_model_l2"]
    combined = defaultdict(list)
    for mn in tree_models:
        for f, v in imp_out.get(mn, {}).items():
            combined[f].append(v)
    imp_out["ensemble_avg"] = dict(sorted(
        {f: float(np.mean(v)) for f, v in combined.items()}.items(), key=lambda kv: -kv[1]))
    (ML_DIR / "feature_importance.json").write_text(
        json.dumps(round_floats(imp_out), indent=2, ensure_ascii=False), encoding="utf-8")

    # ======================= 控制台汇总 =======================
    log("\n" + "=" * 88)
    log(f"{'期间':22s} {'AUC':>6s} {'IC_ens':>7s} {'IC_lgb':>7s} {'IC_xgb':>7s} {'IC_rdg':>7s} "
        f"{'P_hc':>6s} {'基线':>6s} {'覆盖':>6s} {'D10-D1':>9s}")
    for r in fold_records:
        log(f"{r['test_start']}→{r['test_end']}  {r['auc_direction']:6.4f} {r['ic_ensemble']:+7.4f} "
            f"{r['ic_lgbm']:+7.4f} {r['ic_xgbm']:+7.4f} {r['ic_ridge']:+7.4f} "
            f"{r['precision_high_conf']:6.3f} {r['accuracy_l1_all']:6.3f} {r['coverage']:6.1%} "
            f"{r['decile_spread']:+9.5f}")
    log("-" * 88)
    log(f"汇总: 平均AUC={summary['auc_direction']['mean']:.4f} (目标>0.58: "
        f"{'达成' if summary['auc_direction']['target_0.58'] else '未达成'}) | "
        f"平均IC: 集成{summary['ic_spearman']['ensemble']['mean']:+.4f} / "
        f"LGBM{summary['ic_spearman']['lgbm']['mean']:+.4f} / "
        f"XGBM{summary['ic_spearman']['xgbm']['mean']:+.4f} / "
        f"Ridge{summary['ic_spearman']['ridge']['mean']:+.4f}")
    log(f"集成 vs 单模型: 胜过LGBM {summary['ensemble_vs_single']['ensemble_beats_lgbm_folds']}/"
        f"{summary['n_folds']} 折, 胜过XGBM {summary['ensemble_vs_single']['ensemble_beats_xgbm_folds']}/"
        f"{summary['n_folds']} 折, 胜过Ridge {summary['ensemble_vs_single']['ensemble_beats_ridge_folds']}/"
        f"{summary['n_folds']} 折")
    log(f"Meta-labeling: 高置信精确率 {summary['meta_labeling']['mean_high_conf_precision']:.3f} vs "
        f"L1基线 {summary['meta_labeling']['mean_l1_accuracy_baseline']:.3f} "
        f"(提升 {summary['meta_labeling']['precision_lift']:+.3f}), 覆盖率 "
        f"{summary['meta_labeling']['mean_signal_coverage']:.1%}, "
        f"触发信号平均收益 {summary['meta_labeling']['mean_fired_signal_return']:+.5f}")
    log(f"十分位: D10均值 {summary['decile']['mean_top_decile_ret']:+.5f} - D1均值 "
        f"{summary['decile']['mean_bottom_decile_ret']:+.5f} = 价差 {summary['decile']['mean_spread_top_minus_bottom']:+.5f} "
        f"| 单调性 {summary['decile']['mean_monotonicity']:+.3f}")
    log(f"\nTop 10 特征(集成平均): {list(imp_out['ensemble_avg'].items())[:10]}")
    log(f"输出: {ML_DIR / 'predictions.feather'}, {ML_DIR / 'model_performance.json'}, "
        f"{ML_DIR / 'feature_importance.json'}")


if __name__ == "__main__":
    main()

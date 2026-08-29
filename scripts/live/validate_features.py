# -*- coding: utf-8 -*-
"""在线特征管线与离线 feature_matrix.feather 的逐位对齐验证

验证逻辑:
    取 feature_matrix 尾部若干个决策时间戳 (00/08/16 UTC 网格), 对每个时间戳:
      1. 用离线原始 feather 截取该时间戳前 720 根 15m bar (含当根) 构造 mini-panel
         —— 与在线 REST 拉取的窗口完全同构
      2. 走 signals.compute_signal 的同一条特征代码 (02_feature_families)
      3. 逐列对比该时间戳的 123 个特征值 vs feature_matrix 存档值
    全部特征 max|Δ| = 0 (或 < 1e-9) → PASS。任何系统性差异都意味着在线管线
    与回测口径不一致, 禁止上线。

同时校验决策网格: feature_matrix 的 dates[::32] 必须是 00/08/16 UTC。

用法:
    C:/Users/18970/.conda/envs/quant/python.exe scripts/live/validate_features.py
    ... --ts 5        # 验证最近 5 个决策时间戳 (默认 3)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "live"))
ML_DIR = ROOT / "user_data" / "ml_v2"
D = ROOT / "user_data" / "data" / "binance" / "futures"

from signals import BARS_PER_SYMBOL, ML_DIR as _ML, FF, load_models, whitelist_bases  # noqa: E402


def build_window_panel(bases: list[str], ts: pd.Timestamp,
                       bars: int = BARS_PER_SYMBOL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """离线 feather → mini-panel (≤ ts 的最后 bars 根) + BTC 窗口"""
    frames = []
    for b in bases:
        fp = D / f"{b}-15m-futures.feather"
        df = pd.read_feather(fp, columns=["date", "open", "high", "low", "close", "volume"])
        df = df[df["date"] <= ts].tail(bars)
        if len(df):
            frames.append(df.assign(base=b))
    panel = pd.concat(frames, ignore_index=True)
    btc = pd.read_feather(D / "BTC_USDT_USDT-15m-futures.feather",
                          columns=["date", "close"])
    btc = btc[btc["date"] <= ts].tail(bars)
    return panel, btc


def features_at(panel: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    """走在线同一特征代码, 返回 (date, base, 特征...) 长表"""
    tmp = _ML / "_validate_btc_tmp.feather"
    btc[["date", "close"]].to_feather(tmp)
    try:
        cols, _fmap, _meta = FF.build_features(panel, FF.MIN_CS, tmp)
    finally:
        if tmp.exists():
            tmp.unlink()
    feat_df = pd.DataFrame({k: v for k, v in cols.items()})
    feat_df["date"] = panel["date"].to_numpy()
    feat_df["base"] = panel["base"].to_numpy()
    return feat_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=int, default=3, help="验证最近 N 个决策时间戳")
    args = ap.parse_args()

    fm = pd.read_feather(ML_DIR / "feature_matrix.feather")
    meta = {"date", "base", "label_abs", "label_neutral", "label_rank"}
    feat_cols = [c for c in fm.columns if c not in meta]
    fm["date"] = pd.to_datetime(fm["date"], utc=True)

    # --- 决策网格校验: dates[::32] 全部落在 00/08/16 UTC ------------------- #
    dates = sorted(fm["date"].unique())
    grid = pd.DatetimeIndex(dates[::32])
    grid_ok = ((grid.hour % 8 == 0) & (grid.minute == 0)).all()
    print(f"决策网格校验: {len(grid)} 个决策点, 00/08/16 UTC 对齐 = {grid_ok}")
    if not grid_ok:
        print("FAIL: 决策网格与 8h 网格不对齐, 在线锚点需重新推导")
        return 1

    # --- 逐时间戳特征对齐 --------------------------------------------------- #
    # 判定标准 (两层):
    #   1) 特征数值: max|Δ| < 1e-6 (pandas 滚动 std 的在线算法随序列起点有
    #      float32 底噪级累积误差, bb_width/vol 类特征 ~1e-7, 非口径差异)
    #   2) 决策等价 (硬闸门): 用窗口管线 vs feature_matrix 存档分别走
    #      中位数填充 → reg/clf, Top5/Bottom5 集合必须一致, pred/conf_dn 差 < 1e-6
    dts = [t for t in dates if t.hour % 8 == 0 and t.minute == 0][-args.ts:]
    bases = whitelist_bases()
    reg, clf, live_feat_cols, med, _m = load_models()

    assert live_feat_cols == feat_cols, "deploy_features.json 与 feature_matrix 列不一致!"

    all_ok = True
    for ts in dts:
        panel, btc = build_window_panel(bases, ts)
        feat_df = features_at(panel, btc)
        snap = feat_df[feat_df["date"] == ts].set_index("base")
        ref = fm[fm["date"] == ts].set_index("base")

        common = snap.index.intersection(ref.index)
        worst, worst_col, n_bad = 0.0, "", 0
        for c in feat_cols:
            a = snap.loc[common, c].to_numpy(dtype=np.float64)
            b = ref.loc[common, c].to_numpy(dtype=np.float64)
            m = np.isfinite(a) & np.isfinite(b)
            nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
            d = np.abs(a[m] - b[m]).max() if m.any() else 0.0
            if d > 1e-6 or nan_mismatch:
                n_bad += 1
                if d > worst:
                    worst, worst_col = d, c
            if nan_mismatch:
                print(f"  [WARN] {c}: NaN 模式不一致 {nan_mismatch} 行")
        feat_ok = n_bad == 0 and worst <= 1e-6

        # --- 决策等价 (硬闸门) --------------------------------------------- #
        med_s = pd.Series(med)
        Xw = snap.loc[common, feat_cols].fillna(med_s).to_numpy(dtype=np.float32)
        Xr = ref.loc[common, feat_cols].fillna(med_s).to_numpy(dtype=np.float32)
        pw, pr = reg.predict(Xw), reg.predict(Xr)
        cw, cr = clf.predict_proba(Xw)[:, 0], clf.predict_proba(Xr)[:, 0]

        rw = pd.Series(pw, index=common).sort_values(ascending=False)
        rr = pd.Series(pr, index=common).sort_values(ascending=False)
        top5_eq = set(rw.index[:5]) == set(rr.index[:5])
        bot5_eq = set(rw.index[-5:]) == set(rr.index[-5:])
        # conf 门槛下的空头集合 (Bottom5 中 conf_dn > 0.5 者)
        common_list = list(common)
        order_w, order_r = rw.index[-5:], rr.index[-5:]
        shorts_w = {b for b in order_w if cw[common_list.index(b)] > 0.5}
        shorts_r = {b for b in order_r if cr[common_list.index(b)] > 0.5}
        shorts_eq = shorts_w == shorts_r
        d_pred = np.abs(pw - pr).max()
        d_conf = np.abs(cw - cr).max()

        decide_ok = top5_eq and bot5_eq and shorts_eq and d_pred < 1e-6 and d_conf < 1e-6
        status = "PASS" if (feat_ok and decide_ok) else "FAIL"
        if not (feat_ok and decide_ok):
            all_ok = False
        print(f"{ts}  coins={len(common):>2d}  特征偏差列={n_bad} max|Δ|={worst:.2e} ({worst_col})  "
              f"Top5={'=' if top5_eq else 'X'} Bot5={'=' if bot5_eq else 'X'} "
              f"空头集={'=' if shorts_eq else 'X'} max|Δpred|={d_pred:.1e} max|Δconf|={d_conf:.1e}  [{status}]")

    print("\n" + ("ALL PASS — 在线特征管线与回测决策等价" if all_ok
                 else "FAIL — 在线管线与回测口径不一致, 禁止上线"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

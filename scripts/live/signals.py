# -*- coding: utf-8 -*-
"""ML v2 横截面策略 — 在线信号模块 (dry-run 与实盘共用)

信号流水线（与离线研究管线逐位对齐）:
    1. 拉取 27 币 + BTC 的 15m K线 (Binance Futures REST, 只用已收盘 bar)
    2. 构造 mini-panel → 复用 02_feature_families.build_features() 计算全部特征
       (研究期 BTC 不在 27 币 panel 内 → BTC 以临时 feather 传入, 与回测同路径)
    3. NaN 按训练集中位数填充 → deploy_model.pkl 排序分 + deploy_clf.pkl conf_dn
    4. pred 降序排名 → 决策 (ranked 全量列表, conf_dn 字典)

决策语义 (与 07 引擎 / NT 回测 build_decisions 完全一致):
    - 决策网格: bar open_time 为 00:00/08:00/16:00 UTC (8h × 32 根 15m bar)
    - 快照有效币数 < 10 → 跳过该决策 (与 build_decisions 相同)
    - 空头门槛: conf_dn > 0.50 (deploy_meta.json short_conf_threshold)

特征无需的未来数据: 全部特征仅用 t 及以前; 最深回看链 ret_288_zscore /
rv_288_zscore = 384 根收盘价 (~4 天) → 拉取 BARS_PER_SYMBOL=720 根留足余量。

用法:
    python scripts/live/signals.py                 # 拉实时K线, 打印当前决策快照
    python scripts/live/signals.py --json out.json # 同上并写决策 JSON (调试用)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = ROOT / "user_data" / "ml_v2"
CONFIG_PATH = ROOT / "config.json"

BINANCE_FAPI = "https://fapi.binance.com"
BARS_PER_SYMBOL = 500   # 15m bars (~5.2 天); 最深回看链 384 根 + 余量, kline 权重 2 (720 根为 5)
MIN_SNAP_COINS = 10     # 快照有效币数下限 (与 07/NT build_decisions 一致)


def _load_feature_builder():
    """导入 scripts/ml_v2/02_feature_families.py (数字开头文件名, 只能按路径加载)"""
    path = ROOT / "scripts" / "ml_v2" / "02_feature_families.py"
    spec = importlib.util.spec_from_file_location("feat_families_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FF = _load_feature_builder()


# --------------------------------------------------------------------------- #
# 部署模型
# --------------------------------------------------------------------------- #
def load_models(ml_dir: Path = ML_DIR):
    """返回 (reg, clf, feat_cols, fill_medians, meta)"""
    import joblib

    reg = joblib.load(ml_dir / "deploy_model.pkl")
    clf = joblib.load(ml_dir / "deploy_clf.pkl")
    spec = json.loads((ml_dir / "deploy_features.json").read_text(encoding="utf-8"))
    meta = json.loads((ml_dir / "deploy_meta.json").read_text(encoding="utf-8"))
    return reg, clf, spec["feature_cols"], spec["fill_medians"], meta


def whitelist_bases(config_path: Path = CONFIG_PATH) -> list[str]:
    """返回研究管线的 base 命名 (feather pair 前缀, 如 "ETH_USDT_USDT").

    与 feature_matrix / wf_preds_v2 / NT base2sym 的 base 完全同一约定。
    """
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return [p.split("/")[0] + "_USDT_USDT" for p in cfg["exchange"]["pair_whitelist"]]


def base_to_symbol(base: str) -> str:
    """feather pair 前缀 → Binance 合约符号 ("ETH_USDT_USDT" → "ETHUSDT")"""
    return base.split("_")[0] + "USDT"


# --------------------------------------------------------------------------- #
# K线获取
# --------------------------------------------------------------------------- #
def fetch_klines(symbol: str, interval: str = "15m", limit: int = BARS_PER_SYMBOL,
                 proxy: str | None = None, base_url: str = BINANCE_FAPI) -> pd.DataFrame:
    """Binance U 本位合约 K线 → DataFrame(date, open, high, low, close, volume).

    date = bar open_time (UTC, 与离线 feather 同语义); 最后一根未收盘 bar 丢弃。
    """
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
    proxies = {"http": proxy, "https": proxy} if proxy else None
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(f"{base_url}/fapi/v1/klines", params=params,
                             proxies=proxies, timeout=15)
            if r.status_code == 429:
                # 共享代理出口 IP 常见限频: 按 Retry-After 退避
                wait = float(r.headers.get("Retry-After", 60))
                print(f"[signals] {symbol} 429 限频, 退避 {wait:.0f}s "
                      f"(attempt {attempt + 1}/5)", flush=True)
                time.sleep(min(wait, 90))
                last_err = RuntimeError("429 Too Many Requests")
                continue
            r.raise_for_status()
            rows = r.json()
            df = pd.DataFrame(rows, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qv", "n", "tb", "tq", "i",
            ])
            df["date"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = df[c].astype(float)
            df = df.iloc[:-1]  # 丢掉未收盘的最后一根
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:  # noqa: BLE001 — 网络重试
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch_klines({symbol}) 5 次失败: {last_err}")


def fetch_all(bases: list[str], proxy: str | None = None,
              limit: int = BARS_PER_SYMBOL,
              pace: float = 1.0) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """拉全部交易币种 + BTC 基准. 返回 (klines_by_base, btc_klines).

    pace: 逐请求间隔秒 — 控制请求权重增速, 避免触发 Binance IP 限频 (-1003 封禁)
    """
    out = {}
    for i, b in enumerate(bases):
        if i:
            time.sleep(pace)
        out[b] = fetch_klines(base_to_symbol(b), proxy=proxy, limit=limit)
    time.sleep(pace)
    btc = fetch_klines("BTCUSDT", proxy=proxy, limit=limit)
    return out, btc


# --------------------------------------------------------------------------- #
# 信号计算
# --------------------------------------------------------------------------- #
def compute_signal(klines_by_base: dict[str, pd.DataFrame], btc_klines: pd.DataFrame,
                   reg, clf, feat_cols: list[str], fill_medians: dict[str, float],
                   min_snap: int = MIN_SNAP_COINS) -> dict | None:
    """K线 → 决策快照. 返回 None 表示快照不足 (币数 < min_snap).

    返回: {"date", "ranked": [base...], "conf_dn": {base: float},
           "pred": {base: float}, "n_coins", "n_missing"}
    """
    frames = []
    for base, df in klines_by_base.items():
        if df is None or len(df) == 0:
            continue
        f = df.copy()
        f["base"] = base
        frames.append(f)
    if not frames:
        return None
    panel = pd.concat(frames, ignore_index=True)
    panel = FF.prepare_panel(panel)

    # BTC 以临时 feather 传入 (与回测 --btc-feather 回退路径一致; BTC 不入 panel,
    # 否则 cs_rank 会在 28 个币上算, 与研究期 27 币口径不符)
    tmp = None
    btc_feather: Path | None = None
    if btc_klines is not None and len(btc_klines):
        tmp = ML_DIR / "_live_btc_tmp.feather"
        btc_klines[["date", "close"]].to_feather(tmp)
        btc_feather = tmp

    try:
        cols, _fmap, _meta = FF.build_features(panel, FF.MIN_CS, btc_feather)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()

    feat_df = pd.DataFrame({k: v for k, v in cols.items()})
    feat_df["date"] = panel["date"].to_numpy()
    feat_df["base"] = panel["base"].to_numpy()

    snap_ts = feat_df["date"].max()
    snap = feat_df[feat_df["date"] == snap_ts].copy()
    n_missing = len(klines_by_base) - len(snap)
    if len(snap) < min_snap:
        print(f"[signals] 快照币数 {len(snap)} < {min_snap}, 跳过决策 @ {snap_ts}")
        return None

    med = pd.Series(fill_medians)
    X = snap[feat_cols].fillna(med).to_numpy(dtype=np.float32)
    snap["pred"] = reg.predict(X)
    snap["conf_dn"] = clf.predict_proba(X)[:, 0]
    snap = snap.sort_values("pred", ascending=False)

    return {
        "date": pd.Timestamp(snap_ts).isoformat(),
        "ranked": snap["base"].tolist(),
        "conf_dn": {b: float(c) for b, c in zip(snap["base"], snap["conf_dn"])},
        "pred": {b: float(p) for b, p in zip(snap["base"], snap["pred"])},
        "n_coins": int(len(snap)),
        "n_missing": int(n_missing),
    }


def signal_snapshot(proxy: str | None = None) -> dict | None:
    """便捷入口: 拉实时K线 → 决策快照 (dry-run 节点与调试共用)"""
    reg, clf, feat_cols, med, _meta = load_models()
    bases = whitelist_bases()
    kl, btc = fetch_all(bases, proxy=proxy)
    return compute_signal(kl, btc, reg, clf, feat_cols, med)


# --------------------------------------------------------------------------- #
# CLI 调试
# --------------------------------------------------------------------------- #
def main() -> int:
    proxy = None
    for a in sys.argv[1:]:
        if a.startswith("--proxy="):
            proxy = a.split("=", 1)[1]
    t0 = time.time()
    snap = signal_snapshot(proxy=proxy)
    if snap is None:
        return 1
    n = len(snap["ranked"])
    print(f"决策快照 @ {snap['date']}  coins={snap['n_coins']} (missing={snap['n_missing']}) "
          f"耗时 {time.time() - t0:.1f}s")
    print(f"{'rank':>4s} {'base':<10s} {'pred':>9s} {'conf_dn':>8s}")
    for i, b in enumerate(snap["ranked"]):
        mark = ""
        if i < 3:
            mark = "LONG"
        elif i >= n - 3:
            mark = "SHORT?" if snap["conf_dn"][b] > 0.50 else "(conf)"
        print(f"{i:>4d} {b:<10s} {snap['pred'][b]:>+9.5f} {snap['conf_dn'][b]:>8.3f}  {mark}")
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--json=")), None)
    if out:
        Path(out).write_text(json.dumps(snap, indent=1), encoding="utf-8")
        print(f"已写 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

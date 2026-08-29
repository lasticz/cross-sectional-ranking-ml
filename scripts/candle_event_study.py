# -*- coding: utf-8 -*-
"""K线连形态事件研究: 连续 N 根同色线后入场，止盈/超时离场（"赚了就跑"）。

在 27 币 × 15m K线上做顺序（非重叠）模拟:
  多头: 连续 N 根红线后，下一根开盘买入
  空头: 连续 N 根绿线后，下一根开盘卖出
  离场: 触及止盈(相对入场价 +tp) 或持有 M 根后收盘离场; 可选止损
  费用: maker 往返 0.04% 名义（含在每笔净收益中）
train = 2022-2024, test = 2025+，分别统计。

用法: python scripts/candle_event_study.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"

FEE_RT = 0.0004     # maker-maker 往返
MAX_BARS = 12       # 最长持有 12 根 15m = 3 小时
SPLIT = "2025-01-01"

CONFIGS = []
for n_candles in (3, 4):
    for tp in (0.002, 0.003, 0.005, None):   # 止盈 0.2%/0.3%/0.5%/不止盈
        for sl in (0.005, None):             # 止损 0.5%/无
            CONFIGS.append((n_candles, tp, sl))


def load_pairs():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    out = {}
    for p in cfg["exchange"]["pair_whitelist"]:
        base = p.split("/")[0]
        f = D / f"{base}_USDT_USDT-15m-futures.feather"
        if f.exists():
            df = pd.read_feather(f)
            out[base] = df
    return out


def simulate(df, n_candles, tp, sl, direction, split=None):
    """顺序模拟一个币一段数据的非重叠交易，返回每笔净收益率(小数)列表。"""
    if split == "train":
        df = df[df["date"] < SPLIT]
    elif split == "test":
        df = df[df["date"] >= SPLIT]
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    bear = c < o  # 红线
    bull = c > o  # 绿线
    sig = np.ones(len(df), dtype=bool)
    for k in range(n_candles):
        sig &= np.roll(bear if direction == "long" else bull, k)
    sig[: n_candles] = False

    trades = []
    i, n = 0, len(df)
    while i < n - 1:
        if sig[i]:
            entry_i = i + 1
            entry = o[entry_i]
            exit_ret = None
            for j in range(entry_i, min(entry_i + MAX_BARS, n)):
                lo, hi, close = l[j], h[j], c[j]
                if direction == "long":
                    if sl is not None and lo <= entry * (1 - sl):
                        exit_ret = -sl
                        break
                    if tp is not None and hi >= entry * (1 + tp):
                        exit_ret = tp
                        break
                    exit_ret = close / entry - 1
                else:
                    if sl is not None and hi >= entry * (1 + sl):
                        exit_ret = -sl
                        break
                    if tp is not None and lo <= entry * (1 - tp):
                        exit_ret = tp
                        break
                    exit_ret = entry / close - 1
            if exit_ret is not None:
                trades.append(exit_ret - FEE_RT)
            i = entry_i + MAX_BARS  # 顺序非重叠: 直接跳过持有期
        else:
            i += 1
    return trades


def summarize(all_trades):
    a = np.array(all_trades)
    if len(a) == 0:
        return None
    tstat = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 2 else 0
    return {
        "trades": len(a),
        "win%": round(float((a > 0).mean()) * 100, 1),
        "avg_bps": round(float(a.mean()) * 10000, 1),
        "total_bps": round(float(a.sum()) * 10000, 0),
        "t_stat": round(float(tstat), 2),
    }


def main():
    pairs = load_pairs()
    print(f"币数: {len(pairs)}  费用: 往返 {FEE_RT*100:.2f}%  最长持有: {MAX_BARS} 根15m\n")
    rows = []
    for direction, dir_name in (("long", "连红后买(反转)"), ("short", "连绿后卖(反转)")):
        for n_candles, tp, sl in CONFIGS:
            res = {}
            for split in ("train", "test"):
                tr = []
                for df in pairs.values():
                    tr.extend(simulate(df, n_candles, tp, sl, direction, split))
                res[split] = summarize(tr)
            rows.append({
                "dir": dir_name, "N": n_candles,
                "tp": "无" if tp is None else f"{tp*100:.1f}%",
                "sl": "无" if sl is None else f"{sl*100:.1f}%",
                **{f"{k}@{s}": res[s][k] for s in ("train", "test") for k in ("trades", "win%", "avg_bps", "t_stat")},
            })
    cols = ["dir", "N", "tp", "sl",
            "trades@train", "win%@train", "avg_bps@train", "t_stat@train",
            "trades@test", "win%@test", "avg_bps@test", "t_stat@test"]
    out = pd.DataFrame(rows)[cols].sort_values("avg_bps@train", ascending=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    out.to_csv(ROOT / "user_data" / "walkforward_results" / "candle_study.csv", index=False)


if __name__ == "__main__":
    main()

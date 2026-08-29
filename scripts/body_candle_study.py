# -*- coding: utf-8 -*-
"""实体阴线回调策略事件研究（15m）

用户假设: EMA200 上方（多头大背景）出现 3 根连续阴线，且阴线实体占比高（"没影线的实线"），
         下一根开盘做多，到点止盈，配止损。
扫描: 实体/振幅阈值 {0.6, 0.75} × 止盈 {0.3%, 0.5%, 1%, 无} × 止损 {0.5%, 1%, 无}
费用: maker 往返 0.04%；最长持有 16 根（4h）；顺序非重叠。
用法: python scripts/body_candle_study.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
FEE = 0.0004
MAX_BARS = 16
SPLIT = "2025-01-01"


def pair_arrays(base):
    df = pd.read_feather(D / f"{base}_USDT_USDT-15m-futures.feather")
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    ema = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy()
    rng = h - l
    body = np.abs(c - o)
    ratio = np.where(rng > 0, body / np.where(rng == 0, np.nan, rng), 0.0)
    return df["date"], o, h, l, c, ema, ratio


def sim(base, body_th, tp, sl, split):
    dates, o, h, l, c, ema, ratio = pair_arrays(base)
    cut = pd.Timestamp(SPLIT, tz="UTC")
    mask = (dates < cut).to_numpy() if split == "train" else (dates >= cut).to_numpy()
    idx = np.where(mask)[0]
    red_full = (c < o) & (ratio >= body_th)
    n = len(c)
    trades, k, n_idx = [], 0, len(idx)
    while k < n_idx:
        i = idx[k]
        if i >= 2 and i + 1 < n and red_full[i] and red_full[i - 1] and red_full[i - 2] and c[i] > ema[i]:
            entry = o[i + 1]
            ret = None
            for j in range(i + 1, min(i + 1 + MAX_BARS, n)):
                if sl is not None and l[j] <= entry * (1 - sl):
                    ret = -sl
                    break
                if tp is not None and h[j] >= entry * (1 + tp):
                    ret = tp
                    break
                ret = c[j] / entry - 1
            if ret is not None:
                trades.append(ret - FEE)
            while k < n_idx and idx[k] <= i + MAX_BARS:
                k += 1
        else:
            k += 1
    return trades


def main():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    wl = [p.split("/")[0] for p in cfg["exchange"]["pair_whitelist"]]
    pairs = [b for b in wl if (D / f"{b}_USDT_USDT-15m-futures.feather").exists()]

    rows = []
    for body_th in (0.6, 0.75):
        for tp in (0.003, 0.005, 0.01, None):
            for sl in (0.005, 0.01, None):
                r = {
                    "body>=": body_th,
                    "tp": "无" if tp is None else f"{tp*100:.1f}%",
                    "sl": "无" if sl is None else f"{sl*100:.1f}%",
                }
                for split in ("train", "test"):
                    tr = []
                    for b in pairs:
                        tr.extend(sim(b, body_th, tp, sl, split))
                    a = np.array(tr)
                    if len(a) > 3:
                        t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
                        r[split] = (f"{len(a)}笔 胜率{(a>0).mean()*100:.0f}% "
                                    f"均值{a.mean()*1e4:+.1f}bps t={t:.1f}")
                    else:
                        r[split] = f"{len(a)}笔"
                rows.append(r)
    out = pd.DataFrame(rows).sort_values("body>=", ascending=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    out.to_csv(ROOT / "user_data" / "walkforward_results" / "body_candle_study.csv", index=False)


if __name__ == "__main__":
    main()

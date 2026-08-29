# -*- coding: utf-8 -*-
"""5m 均值回归事件研究

信号: close < BB下轨(20, std) & RSI<th & close > EMA200 -> 做多(对称做空)
入场: 下一根开盘(现实口径); 离场: 收盘回到布林中轨 或 超时 N 根
费用: maker 往返 4bps。train 2023-24 / test 2025+。
用法: python scripts/mr5m_study.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
FEE = 0.0004
TIMEOUT = 12  # 根 (1小时)


def load_all():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    out = {}
    for p in cfg["exchange"]["pair_whitelist"]:
        b = p.split("/")[0]
        f = D / f"{b}_USDT_USDT-5m-futures.feather"
        if f.exists():
            out[b] = pd.read_feather(f)
    return out


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    return 100 - 100 / (1 + ru / np.where(rd == 0, 1e-12, rd))


def sim_pair(df, bb_std, rsi_th, direction):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(c)
    mid = pd.Series(c).rolling(20).mean().to_numpy()
    sd = pd.Series(c).rolling(20).std().to_numpy()
    lower = mid - bb_std * sd
    upper = mid + bb_std * sd
    r = rsi(c)
    ema = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy()

    if direction == "long":
        sig = (c < lower) & (r < rsi_th) & (c > ema)
    else:
        sig = (c > upper) & (r > 100 - rsi_th) & (c < ema)

    trades = []
    i, idx = 0, 0
    sig_idx = np.where(sig)[0]
    skip_until = -1
    for i in sig_idx:
        if i + 1 >= n or i <= skip_until:
            continue
        entry = o[i + 1]
        ret = None
        for j in range(i + 1, min(i + 1 + TIMEOUT, n)):
            # 先判中轨回归(收盘价), 超时在最后一根收盘离场
            if direction == "long" and c[j] >= mid[j]:
                ret = c[j] / entry - 1
                break
            if direction == "short" and c[j] <= mid[j]:
                ret = entry / c[j] - 1
                break
        if ret is None:
            j = min(i + TIMEOUT, n - 1)
            ret = (c[j] / entry - 1) if direction == "long" else (entry / c[j] - 1)
        trades.append((df["date"].iloc[i], ret - FEE))
        skip_until = i + TIMEOUT  # 顺序非重叠
    return trades


def main():
    data = load_all()
    print(f"币数 {len(data)}, 超时 {TIMEOUT}根(1h), 费 {FEE*100:.2f}%/往返\n")
    t0 = time.time()
    for bb_std in (2.0, 2.5):
        for rsi_th in (25, 30):
            for direction in ("long", "short"):
                tr, te = [], []
                for b, df in data.items():
                    for ts, ret in sim_pair(df, bb_std, rsi_th, direction):
                        (tr if ts < SPLIT else te).append(ret)
                a, b_ = np.array(tr), np.array(te)
                def st(x):
                    if len(x) < 50:
                        return f"n={len(x)}"
                    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
                    return f"n={len(x):6d} {x.mean()*1e4:+6.1f}bps 胜率{(x>0).mean()*100:.0f}% t={t:+5.1f} 天/笔={365*len(x)/2/len(data):.1f}"
                print(f"BB{bb_std} RSI{'<'+str(rsi_th) if direction=='long' else '>'+str(100-rsi_th)} {direction:5s}: "
                      f"train {st(a)} | test {st(b_)}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

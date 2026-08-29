# -*- coding: utf-8 -*-
"""新币上线事件研究: 上线后 D+1..D+30 的价格结构。

对每个 2023-01 后上线的永续（币类），以首个日收盘为基准，
统计前推 k 日的累计收益（原始 & 相对 BTC 超额），按上线年份分群组看稳定性。

关注形态:
  A. 上线首日后的"阴跌"（short-the-bleed）: D+1/3/5 进空，持有 10/20/30 天
  B. 上线初期动量: D0→D+3
用法: python scripts/listing_study.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
HORIZONS = (1, 3, 5, 10, 20, 30)


def daily_close(base: str) -> pd.Series | None:
    f = D / f"{base}_USDT_USDT-1h-futures.feather"
    if not f.exists():
        return None
    df = pd.read_feather(f)
    s = df.set_index("date")["close"].resample("1D").last().dropna()
    return s[s.index >= s.index[0]]


def cum_path(s: pd.Series, k: int) -> float | None:
    """相对首日收盘的前推 k 日累计收益。"""
    if len(s) <= k:
        return None
    return float(s.iloc[k] / s.iloc[0] - 1)


def main():
    listings = json.loads((ROOT / "user_data" / "listings.json").read_text(encoding="utf-8"))
    btc = daily_close("BTC")
    btc_ret = btc.pct_change()

    rows = []
    for item in listings:
        s = daily_close(item["base"])
        if s is None or len(s) < 8:
            continue
        d0 = s.index[0]
        row = {"base": item["base"], "onboard": item["onboard"], "cohort": str(d0.year),
               "days": len(s)}
        for k in HORIZONS:
            r = cum_path(s, k)
            row[f"r{k}"] = r
            # 超额: 同期 BTC 收益
            if r is not None and d0 + pd.Timedelta(days=k) <= btc.index[-1]:
                loc = btc.index.get_indexer([d0 + pd.Timedelta(days=k)])[0]
                if loc >= 0 and not np.isnan(btc_ret.iloc[loc]):
                    row[f"e{k}"] = r - float(btc_ret.iloc[loc])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "user_data" / "walkforward_results" / "listing_paths.csv", index=False)
    print(f"可用样本: {len(df)} 个新币\n")

    print("== 各上线年份: 前推k日累计收益 中位数% (n) ==")
    for cohort, g in df.groupby("cohort"):
        cells = []
        for k in HORIZONS:
            col = g[f"r{k}"].dropna()
            cells.append(f"D+{k}:{col.median()*100:+.1f}%" if len(col) else f"D+{k}:--")
        print(f"  {cohort} (n={len(g)}): " + " ".join(cells))

    print("\n== 交易形态统计（原始收益，正=做空赚钱的方向取负号看） ==")
    for k in (3, 5, 10, 20, 30):
        col = df[f"r{k}"].dropna()
        if not len(col):
            continue
        short_edge = -col  # 做空收益 = -价格收益
        # 简单胜率与中位数（费前；多日持有费率可忽略）
        print(f"  D0进空持有{k:>2}天: n={len(col):3d} 中位数{short_edge.median()*100:+.2f}% "
              f"均值{short_edge.mean()*100:+.2f}% 胜率{(short_edge > 0).mean()*100:.0f}%")
    print()
    for k in (3, 5, 10, 20, 30):
        col = df[f"e{k}"].dropna()  # 超额做空
        if not len(col):
            continue
        print(f"  D0进空持有{k:>2}天(对冲BTC): n={len(col):3d} 中位数{-col.median()*100:+.2f}% "
              f"胜率{(-col > 0).mean()*100:.0f}%")

    print("\n== 分年: D+10 做空中位数（看形态稳定性） ==")
    for cohort, g in df.groupby("cohort"):
        col = g["r10"].dropna()
        if len(col):
            print(f"  {cohort}: n={len(col)} 中位数{-col.median()*100:+.2f}% 胜率{(-col>0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()

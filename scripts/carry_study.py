# -*- coding: utf-8 -*-
"""资金费率 Carry 组合研究（8h 轮换）

每期(8h)排名: 做多费率最负 K 个（空头付租给多头），做空费率最正 K 个（多头付租给空头）。
每期收益 = 持仓期价格变动 ± 收到/付出的资金费。分解"租金收益"和"价格拖累"。
train = 2025-01 前, test = 之后。费率数据: 27 币 8h 序列。
用法: python scripts/carry_study.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")


def load_all():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    funds, prices = {}, {}
    for p in cfg["exchange"]["pair_whitelist"]:
        b = p.split("/")[0]
        ff, kf = D / f"{b}_USDT_USDT-1h-funding_rate.feather", D / f"{b}_USDT_USDT-1h-futures.feather"
        if ff.exists() and kf.exists():
            s = pd.read_feather(ff).set_index("date")["open"]
            s.index = s.index.floor("8h")  # 币安结算时间带毫秒抖动, 对齐 8h 网格
            funds[b] = s[~s.index.duplicated()]
            prices[b] = pd.read_feather(kf).set_index("date")["close"]
    return funds, prices


def main():
    funds, prices = load_all()
    F = pd.DataFrame(funds)           # 8h 网格, 当期费率
    P = pd.DataFrame(prices).resample("8h").last()

    for K in (2, 3):
        long_pnl, short_pnl, carry_l, carry_s = [], [], [], []
        dates = []
        for t in range(len(F) - 1):
            f_now = F.iloc[t].dropna()
            if len(f_now) < 10:
                continue
            # 用当期费率排名 → 持有到下一期结算
            f_next = F.iloc[t + 1]
            p_now, p_next = P.loc[F.index[t]], P.loc[F.index[t + 1]]
            bottom = f_now.nsmallest(K).index   # 最负 → 做多收租
            top = f_now.nlargest(K).index       # 最正 → 做空收租

            lp, lc, sp, sc = [], [], [], []
            for b in bottom:
                if np.isfinite(f_next.get(b, np.nan)) and np.isfinite(p_now.get(b)) and np.isfinite(p_next.get(b)):
                    ret = p_next[b] / p_now[b] - 1
                    lp.append(ret - f_next[b])          # 多头资金费盈亏 = -f (f<0 时空头付租)
                    lc.append(-f_next[b])
            for b in top:
                if np.isfinite(f_next.get(b, np.nan)) and np.isfinite(p_now.get(b)) and np.isfinite(p_next.get(b)):
                    ret = p_next[b] / p_now[b] - 1
                    sp.append(-ret + f_next[b])         # 空头资金费盈亏 = +f (f>0 时多头付租)
                    sc.append(f_next[b])
            if lp and sp:
                long_pnl.append(np.mean(lp)); carry_l.append(np.mean(lc))
                short_pnl.append(np.mean(sp)); carry_s.append(np.mean(sc))
                dates.append(F.index[t + 1])

        df = pd.DataFrame({"long": long_pnl, "short": short_pnl,
                           "carry_l": carry_l, "carry_s": carry_s}, index=pd.DatetimeIndex(dates))
        for name, split in (("train 22-24", df.index < SPLIT), ("test 25+", df.index >= SPLIT)):
            d = df[split]
            n = len(d)
            if n == 0:
                continue
            tot = d["long"] + d["short"]
            print(f"K={K} {name}: 期数={n}")
            print(f"  多腿: {d['long'].mean()*1e4:+.2f}bps/期 (租金 {d['carry_l'].mean()*1e4:+.2f}) "
                  f"年化 {d['long'].mean()*3*365*100:+.1f}%")
            print(f"  空腿: {d['short'].mean()*1e4:+.2f}bps/期 (租金 {d['carry_s'].mean()*1e4:+.2f}) "
                  f"年化 {d['short'].mean()*3*365*100:+.1f}%")
            print(f"  合计: {tot.mean()*1e4:+.2f}bps/期, 年化 {tot.mean()*3*365*100:+.1f}%, "
                  f"累计 {((1+tot).prod()-1)*100:+.1f}%, 期胜率 {(tot>0).mean()*100:.0f}%")
        print()


if __name__ == "__main__":
    main()

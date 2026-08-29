# -*- coding: utf-8 -*-
"""异常扫描: 在 27 币原始数据里找简单、少人注意的规律性，全部 train/test 分段验证。

扫描项:
  1. 日内时段效应(UTC): 各小时平均收益，train 选出的好时段在 test 是否延续
  2. 周内效应: 各星期平均收益
  3. 资金费结算窗漂移: 结算(00/08/16 UTC)前 3h / 后 3h 收益，按费率高低分桶
  4. 长影线拒绝: 20根K线新低+下影线>=60%振幅 → 做多，前瞻 4 根(1h)
  5. BTC 领先传导: BTC 15m 大涨/大跌后，山寨下一根的平均收益
用法: python scripts/anomaly_scan.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
FEE_RT = 0.0004


def load(base, tf):
    f = D / f"{base}_USDT_USDT-{tf}-futures.feather"
    return pd.read_feather(f) if f.exists() else None


def pairs():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    return [p.split("/")[0] for p in cfg["exchange"]["pair_whitelist"]]


def split_stats(x_train, x_test, name, unit="bps"):
    k = 10000 if unit == "bps" else 100
    def st(a):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < 5:
            return "n<5"
        t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 2 else 0
        return f"n={len(a):.0f} 均值{a.mean()*k:+.1f}{unit} t={t:+.1f}"
    print(f"  {name:34s} | train: {st(x_train)} | test: {st(x_test)}")


def scan_hour_of_day():
    print("\n[1] 日内时段效应 (1h, 全币池)")
    rows = []
    for b in pairs():
        df = load(b, "1h")
        if df is None:
            continue
        r = df["close"].pct_change()
        rows.append(pd.DataFrame({"hour": df["date"].dt.hour, "ret": r, "is_train": df["date"] < SPLIT}))
    big = pd.concat(rows)
    tr = big[big["is_train"]].groupby("hour")["ret"].mean()
    te = big[~big["is_train"]].groupby("hour")["ret"].mean()
    tab = pd.DataFrame({"train_bps": tr * 1e4, "test_bps": te * 1e4}).round(1)
    print(tab.to_string())
    top = tr.nlargest(3).index.tolist()
    bot = tr.nsmallest(3).index.tolist()
    # 用 train 选出的时段在 test 复合持有（每小时一笔，扣费）
    for hours, side in ((top, "long"), (bot, "short")):
        sel = big[~big["is_train"] & big["hour"].isin(hours)]
        r = sel["ret"].to_numpy()
        r = r[np.isfinite(r)]
        net = (r if side == "long" else -r) - FEE_RT
        print(f"  test段 持有{hours} {side}: 净均值 {net.mean()*1e4:+.1f}bps/时, n={len(net)}")


def scan_day_of_week():
    print("\n[2] 周内效应 (1h 聚成日)")
    daily = []
    for b in pairs():
        df = load(b, "1h")
        if df is None:
            continue
        s = df.set_index("date")["close"].resample("1D").last().pct_change().dropna()
        daily.append(pd.DataFrame({"wd": s.index.dayofweek, "ret": s.to_numpy(),
                                   "train": s.index < SPLIT}))
    big = pd.concat(daily)
    tr = big[big["train"]].groupby("wd")["ret"].mean() * 1e4
    te = big[~big["train"]].groupby("wd")["ret"].mean() * 1e4
    print(pd.DataFrame({"train_bps": tr.round(1), "test_bps": te.round(1)}).to_string())


def scan_funding_window():
    print("\n[3] 资金费结算窗漂移 (1h, 按结算时费率分桶)")
    for b in ["SOL", "DOGE", "ADA", "AVAX"][:2] + ["SOL"]:  # 少量代表币避免噪声; SOL 重复去重
        pass
    rows_pre, rows_post = [], []
    for b in pairs():
        df = load(b, "1h")
        ff = D / f"{b}_USDT_USDT-1h-funding_rate.feather"
        if df is None or not ff.exists():
            continue
        fund = pd.read_feather(ff).set_index("date")["open"]
        c = df.set_index("date")["close"]
        for settle in fund.index[1:]:
            h = settle.hour
            if h not in (0, 8, 16):
                continue
            try:
                pre = c.loc[settle] / c.loc[settle - pd.Timedelta(hours=3)] - 1
                post = c.loc[settle + pd.Timedelta(hours=3)] / c.loc[settle] - 1
            except KeyError:
                continue
            rows_pre.append((fund.loc[settle], pre, settle < SPLIT))
            rows_post.append((fund.loc[settle], post, settle < SPLIT))
    pre = pd.DataFrame(rows_pre, columns=["f", "ret", "train"]).dropna()
    post = pd.DataFrame(rows_post, columns=["f", "ret", "train"]).dropna()
    hi = pre["f"] >= 0.0001   # 偏正费率
    lo = pre["f"] <= -0.0001  # 偏负费率
    for name, cond, df_ in (("费率偏正·结算前3h", hi, pre), ("费率偏正·结算后3h", hi, post),
                            ("费率偏负·结算前3h", lo, pre), ("费率偏负·结算后3h", lo, post)):
        d = df_[cond]
        split_stats(d[d["train"]]["ret"], d[~d["train"]]["ret"], name)


def scan_wick_rejection():
    print("\n[4] 长影线拒绝 (1h)")
    longs_tr, longs_te, shorts_tr, shorts_te = [], [], [], []
    for b in pairs():
        df = load(b, "1h")
        if df is None:
            continue
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        rng = h - l
        lower = (np.minimum(o, c) - l) / np.where(rng == 0, np.nan, rng)
        upper = (h - np.maximum(o, c)) / np.where(rng == 0, np.nan, rng)
        roll_min = pd.Series(l).rolling(20).min().to_numpy()
        roll_max = pd.Series(h).rolling(20).max().to_numpy()
        sig_l = (lower >= 0.6) & (l <= roll_min * 1.001)
        sig_s = (upper >= 0.6) & (h >= roll_max * 0.999)
        fwd4 = np.full(len(df), np.nan)
        fwd4[:-4] = c[4:] / c[:-4] - 1
        train = (df["date"] < SPLIT).to_numpy()
        longs_tr.extend(fwd4[sig_l & train]); longs_te.extend(fwd4[sig_l & ~train])
        shorts_tr.extend(-fwd4[sig_s & train]); shorts_te.extend(-fwd4[sig_s & ~train])
    split_stats(longs_tr, longs_te, "新低长下影→做多(4h)")
    split_stats(shorts_tr, shorts_te, "新高长上影→做空(4h)")


def scan_btc_leadlag():
    print("\n[5] BTC 领先传导 (15m)")
    btc = load("BTC", "15m").set_index("date")["close"]
    for th in (0.003, 0.005, 0.008):
        ltr, lte, str_, ste = [], [], [], []
        for b in pairs():
            df = load(b, "15m")
            if df is None:
                continue
            idx = df.set_index("date").index
            btc_r = btc.pct_change().reindex(idx)
            nxt = df["close"].pct_change().shift(-1).to_numpy()
            br = btc_r.to_numpy()
            train = (df["date"] < SPLIT).to_numpy()
            ltr.extend(nxt[(br > th) & train]); lte.extend(nxt[(br > th) & ~train])
            str_.extend(-nxt[(br < -th) & train]); ste.extend(-nxt[(br < -th) & ~train])
        split_stats(np.array(ltr) - FEE_RT, np.array(lte) - FEE_RT, f"BTC涨>{th*100:.1f}% → 下根买山寨")
        split_stats(np.array(str_) - FEE_RT, np.array(ste) - FEE_RT, f"BTC跌>{th*100:.1f}% → 下根空山寨")


if __name__ == "__main__":
    scan_hour_of_day()
    scan_day_of_week()
    scan_funding_window()
    scan_wick_rejection()
    scan_btc_leadlag()

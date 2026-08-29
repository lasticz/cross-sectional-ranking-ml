# -*- coding: utf-8 -*-
"""多空完整版 + 空头置信度门槛扫描（验证过的引擎, 杠杆已修复）"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = ROOT / "user_data" / "ml_v2"
D = ROOT / "user_data" / "data" / "binance" / "futures"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
PAIRS = [p.split("/")[0] + "_USDT_USDT" for p in CONFIG["exchange"]["pair_whitelist"]]

FEE = 0.0002
LEV = 3.0
STAKE_PCT = 0.10
MAX_L = 3
MAX_S = 3
BUFFER = 5
REBAL = 32
SL = 0.15
MIN_HOLD = 32


def predict(df, fc):
    months = df["date"].dt.to_period("M")
    ms = sorted(months.unique())
    preds = []
    for i in range(12, len(ms)):
        tr = (months >= ms[i-12]) & (months <= ms[i-1])
        te = months == ms[i]
        if tr.sum() < 10000 or te.sum() < 100: continue
        med = df.loc[tr, fc].median(numeric_only=True)
        X_tr = df.loc[tr, fc].fillna(med).to_numpy(dtype=np.float32)
        y = df.loc[tr, "label_neutral"].to_numpy(dtype=np.float32)
        v = np.isfinite(y)
        clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbose=-1, force_col_wise=True, n_jobs=4)
        clf.fit(X_tr[v], (y[v] > 0).astype(int))
        reg = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.05,
                                 num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbose=-1, force_col_wise=True, n_jobs=4)
        reg.fit(X_tr[v], y[v])
        X_te = df.loc[te, fc].fillna(med).to_numpy(dtype=np.float32)
        proba = clf.predict_proba(X_te)
        preds.append(pd.DataFrame({
            "idx": df.index[te], "date": df.loc[te, "date"].to_numpy(),
            "base": df.loc[te, "base"].to_numpy(), "pred": reg.predict(X_te),
            "conf_up": proba[:, 1], "conf_dn": proba[:, 0]}))
    return pd.concat(preds, ignore_index=True)


def load():
    o, f = {}, {}
    for p in PAIRS:
        fp = D / (p + "-15m-futures.feather")
        if fp.exists(): o[p] = pd.read_feather(fp).set_index("date")[["open","high","low","close"]]
        ff = D / (p + "-1h-funding_rate.feather")
        if ff.exists():
            fr = pd.read_feather(ff).set_index("date")["open"]
            fr.index = fr.index.floor("15min")
            f[p] = fr[~fr.index.duplicated()]
    return o, f


def backtest(pdf, ohlcv, fund, s_th):
    dates = sorted(pdf["date"].unique())
    dds = dates[::REBAL]
    cap = 150.0
    poss = {}
    trades = []
    bc = 0
    f15 = {b: fund[b].reindex(ohlcv[b].index).ffill() for b in fund if b in ohlcv}

    def px(b, d, c="close"):
        if b not in ohlcv: return None
        df = ohlcv[b]
        return df.loc[d, c] if d in df.index else None
    def nopen(b, d):
        if b not in ohlcv: return None
        df = ohlcv[b]
        try:
            i = df.index.get_loc(d)
            return df["open"].iloc[i+1] if i+1 < len(df) else None
        except: return None
    def eq():
        u = 0.0
        for b, p in poss.items():
            c = px(b, dds[0] if not dds else dds[0], "close")
        return cap + u
    def equity_at(d):
        u = 0.0
        for b, p in poss.items():
            c = px(b, d, "close")
            if c:
                if p["side"]=="long": u += (c/p["entry_price"]-1)*p["size"]
                else: u += (1-c/p["entry_price"])*p["size"]
        return cap + u

    for dd in dds:
        bc += 1
        snap = pdf[pdf["date"]==dd].sort_values("pred", ascending=False)
        if len(snap) < 10: continue
        ranked = snap["base"].tolist()
        pmap = snap.set_index("base")
        top = set(ranked[:MAX_L])
        bot = set(ranked[-MAX_S:])
        buf = set(ranked[:BUFFER]) | set(ranked[-BUFFER:])

        # SL
        for b in list(poss):
            p = poss[b]
            lo, hi = px(b, dd, "low"), px(b, dd, "high")
            if lo is None: continue
            hit = False
            if p["side"]=="long" and lo <= p["entry_price"]*(1-SL):
                exit_x = max(lo, p["entry_price"]*(1-SL)); hit = True
            elif p["side"]=="short" and hi >= p["entry_price"]*(1+SL):
                exit_x = min(hi, p["entry_price"]*(1+SL)); hit = True
            if hit:
                g = (exit_x/p["entry_price"]-1)*p["size"] if p["side"]=="long" else (1-exit_x/p["entry_price"])*p["size"]
                fee = p["size"]*FEE; cap += g - fee
                trades.append({"b":b,"s":p["side"],"ret":(g-fee)/p["margin"],"e":"sl","d":dd})
                del poss[b]

        # Exit
        for b in list(poss):
            p = poss[b]
            ex = False
            if p["side"]=="long" and b not in buf: ex = True
            elif p["side"]=="long" and p["hold"]>=MIN_HOLD and b not in top: ex = True
            elif p["side"]=="short" and b not in bot: ex = True
            elif p["side"]=="short" and b in pmap.index and pmap.loc[b,"conf_dn"] < s_th: ex = True
            if ex:
                exit_x = nopen(b, dd) or px(b, dd, "close")
                if exit_x:
                    g = (exit_x/p["entry_price"]-1)*p["size"] if p["side"]=="long" else (1-exit_x/p["entry_price"])*p["size"]
                    fee = p["size"]*FEE; cap += g - fee
                    trades.append({"b":b,"s":p["side"],"ret":(g-fee)/p["margin"],"e":"signal","d":dd})
                    del poss[b]

        # Funding
        if bc % REBAL == 0:
            for b, p in poss.items():
                fr = f15.get(b)
                if fr is not None and dd in fr.index:
                    r = fr.loc[dd]
                    if np.isfinite(r):
                        if p["side"]=="long": cap -= p["size"]*r
                        else: cap += p["size"]*r

        # Open long
        nl = sum(1 for p in poss.values() if p["side"]=="long")
        for b in ranked[:MAX_L]:
            if b in poss or nl >= MAX_L: continue
            ep = nopen(b, dd)
            if not ep or ep <= 0: continue
            eq_v = equity_at(dd)
            m = eq_v * STAKE_PCT; n = m * LEV
            cap -= n * FEE
            poss[b] = {"side":"long","entry_price":ep,"entry_date":dd,"size":n,"margin":m,"hold":0}
            nl += 1

        # Open short (with confidence threshold)
        ns = sum(1 for p in poss.values() if p["side"]=="short")
        for b in ranked[-MAX_S:]:
            if b in poss or ns >= MAX_S: continue
            if b in pmap.index and pmap.loc[b, "conf_dn"] < s_th: continue
            ep = nopen(b, dd)
            if not ep or ep <= 0: continue
            eq_v = equity_at(dd)
            m = eq_v * STAKE_PCT; n = m * LEV
            cap -= n * FEE
            poss[b] = {"side":"short","entry_price":ep,"entry_date":dd,"size":n,"margin":m,"hold":0}
            ns += 1

    return trades, cap


def main():
    t0 = time.time()
    print("=== Walk-Forward 预测（含置信度） ===")
    df = pd.read_feather(ML_DIR / "feature_matrix.feather")
    meta = {"date","base","label_abs","label_neutral","label_rank"}
    fc = [c for c in df.columns if c not in meta]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    pdf = predict(df, fc)
    print(f"预测: {len(pdf)} 行, {time.time()-t0:.0f}s")
    pdf.to_feather(ML_DIR / "wf_preds_v2.feather")

    ohlcv, fund = load()
    print(f"\n{'门槛':>6s} {'总收益%':>10s} {'多PnL':>10s} {'空PnL':>10s} {'交易':>6s} {'空%':>6s}")
    print("-" * 55)
    best_th, best_ret = 0.5, -np.inf
    scan = {}
    for th in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        trs, fc2 = backtest(pdf, ohlcv, fund, th)
        ret = (fc2/150 - 1) * 100
        tdf = pd.DataFrame(trs) if trs else pd.DataFrame()
        lp = tdf[tdf["s"]=="long"]["ret"].sum() if len(tdf) else 0
        sp = tdf[tdf["s"]=="short"]["ret"].sum() if len(tdf) else 0
        spc = len(tdf[tdf["s"]=="short"]) / max(len(tdf),1) * 100
        print(f"{th:>6.2f} {ret:>+10.1f} {lp:>+10.1f} {sp:>+10.1f} {len(tdf):>6d} {spc:>5.1f}%")
        scan[th] = ret
        if ret > best_ret: best_ret, best_th = ret, th

    print(f"\n=== 最佳门槛: {best_th} → {best_ret:+.1f}% ===")
    trs, fc2 = backtest(pdf, ohlcv, fund, best_th)
    tdf = pd.DataFrame(trs)
    tdf["year"] = pd.to_datetime(tdf["d"]).dt.year
    for y, g in tdf.groupby("year"):
        lg = g[g["s"]=="long"]; sg = g[g["s"]=="short"]
        print(f"  {y}: 多{len(lg)}笔({lg['ret'].mean()*100:+.2f}%/笔) 空{len(sg)}笔({sg['ret'].mean()*100:+.2f}%/笔)")

    print(f"\n对比:")
    print(f"  只有多头(选择偏好): +441%")
    print(f"  多空完整(门槛{best_th}): {best_ret:+.1f}%")
    print(f"  理论上限(完美排序): +3747%/年")
    print(f"\n耗时: {time.time()-t0:.0f}s")
    tdf.to_feather(ML_DIR / "full_ls_trades.feather")
    (ML_DIR / "threshold_scan.json").write_text(json.dumps({"best": best_th, "ret": best_ret, "scan": scan}, indent=2))


if __name__ == "__main__":
    main()

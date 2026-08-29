# -*- coding: utf-8 -*-
"""NautilusTrader ML 横截面全期回测 (2023-01 → 2026-08)

复用 nt_xs_ml_backtest.py 的基础设施，扩展到全期间。
分 6 个月窗口运行（避免内存问题），结果拼接。

用法: python scripts/engine/nt_full_backtest.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = ROOT / "user_data" / "ml_v2"
D = ROOT / "user_data" / "data" / "binance" / "futures"

sys.path.insert(0, str(ROOT / "scripts" / "engine"))


def load_predictions():
    """加载 walk-forward 预测"""
    f = ML_DIR / "wf_preds_v2.feather"
    if f.exists():
        df = pd.read_feather(f)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df
    # 如果没有，用 07 的方法重新生成
    print("wf_preds_v2.feather 不存在，需要先运行 07_full_ls_backtest.py")
    sys.exit(1)


def run_period(pred_df, ohlcv, funding, start, end, initial_capital=150.0):
    """在指定时间窗口运行回测（复用 07 的逻辑）"""
    # 过滤预测到指定窗口
    mask = (pred_df["date"] >= start) & (pred_df["date"] < end)
    window_pred = pred_df[mask].copy()
    if len(window_pred) == 0:
        return [], initial_capital, []

    REBAL = 32
    FEE = 0.0002
    LEV = 3.0
    STAKE = 0.10
    MAX_L, MAX_S = 3, 3
    BUFFER = 5
    SL = 0.15
    MIN_HOLD = 32
    S_TH = 0.50

    dates = sorted(window_pred["date"].unique())
    decision_dates = dates[::REBAL]

    capital = initial_capital
    positions = {}
    trades = []
    equity_pts = []
    bar_count = 0

    funding_15m = {}
    for b, fr in funding.items():
        if b in ohlcv:
            funding_15m[b] = fr.reindex(ohlcv[b].index).ffill()

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

    def equity_at(d):
        u = 0.0
        for b, p in positions.items():
            c = px(b, d, "close")
            if c:
                if p["side"]=="long": u += (c/p["entry_price"]-1)*p["size"]
                else: u += (1-c/p["entry_price"])*p["size"]
        return capital + u

    for dd in decision_dates:
        bar_count += 1
        snap = window_pred[window_pred["date"]==dd].sort_values("pred", ascending=False)
        if len(snap) < 10: continue
        ranked = snap["base"].tolist()
        pmap = snap.set_index("base")
        top = set(ranked[:MAX_L])
        bot = set(ranked[-MAX_S:])
        buf = set(ranked[:BUFFER]) | set(ranked[-BUFFER:])

        # SL
        for b in list(poss := positions):
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
                fee = p["size"]*FEE; cap_before = capital; capital += g - fee
                trades.append({"b":b,"s":p["side"],"ret":(g-fee)/p["margin"],"pnl":g-fee,
                               "e":"sl","d":dd,"equity_after":capital})
                del positions[b]

        # Exit
        for b in list(positions):
            p = positions[b]
            ex = False
            if p["side"]=="long" and b not in buf: ex = True
            elif p["side"]=="long" and p.get("hold",0)>=MIN_HOLD and b not in top: ex = True
            elif p["side"]=="short" and b not in bot: ex = True
            elif p["side"]=="short" and b in pmap.index and pmap.loc[b].get("conf_dn",0.5) < S_TH: ex = True
            if ex:
                exit_x = nopen(b, dd) or px(b, dd, "close")
                if exit_x:
                    g = (exit_x/p["entry_price"]-1)*p["size"] if p["side"]=="long" else (1-exit_x/p["entry_price"])*p["size"]
                    fee = p["size"]*FEE; capital += g - fee
                    trades.append({"b":b,"s":p["side"],"ret":(g-fee)/p["margin"],"pnl":g-fee,
                                   "e":"signal","d":dd,"equity_after":capital})
                    del positions[b]

        # Funding
        if bar_count % REBAL == 0:
            for b, p in positions.items():
                fr = funding_15m.get(b)
                if fr is not None and dd in fr.index:
                    r = fr.loc[dd]
                    if np.isfinite(r):
                        if p["side"]=="long": capital -= p["size"]*r
                        else: capital += p["size"]*r

        # Open long
        nl = sum(1 for p in positions.values() if p["side"]=="long")
        for b in ranked[:MAX_L]:
            if b in positions or nl >= MAX_L: continue
            ep = nopen(b, dd)
            if not ep or ep <= 0: continue
            eq_v = equity_at(dd)
            m = eq_v * STAKE; n = m * LEV
            capital -= n * FEE
            positions[b] = {"side":"long","entry_price":ep,"entry_date":dd,"size":n,"margin":m,"hold":0}
            nl += 1

        # Open short
        ns = sum(1 for p in positions.values() if p["side"]=="short")
        for b in ranked[-MAX_S:]:
            if b in positions or ns >= MAX_S: continue
            if b in pmap.index and pmap.loc[b].get("conf_dn",0.5) < S_TH: continue
            ep = nopen(b, dd)
            if not ep or ep <= 0: continue
            eq_v = equity_at(dd)
            m = eq_v * STAKE; n = m * LEV
            capital -= n * FEE
            positions[b] = {"side":"short","entry_price":ep,"entry_date":dd,"size":n,"margin":m,"hold":0}
            ns += 1

        equity_pts.append({"date": dd, "equity": equity_at(dd), "n_pos": len(positions)})

    # 强平剩余仓位
    for b in list(positions):
        p = positions.pop(b)
        exit_x = px(b, decision_dates[-1] if decision_dates else all_dates[-1], "close")
        if exit_x:
            g = (exit_x/p["entry_price"]-1)*p["size"] if p["side"]=="long" else (1-exit_x/p["entry_price"])*p["size"]
            fee = p["size"]*FEE; capital += g - fee
            trades.append({"b":b,"s":p["side"],"ret":(g-fee)/p["margin"],"pnl":g-fee,
                           "e":"end","d":end,"equity_after":capital})

    return trades, capital, equity_pts


def main():
    t0 = time.time()

    # 加载预测
    print("加载预测...")
    pred_df = load_predictions()
    print(f"  {len(pred_df)} 行, {pred_df['date'].min()} → {pred_df['date'].max()}")

    # 加载 OHLCV 和 funding
    print("加载 K线和资金费...")
    ohlcv = {}
    funding = {}
    for pair in PAIRS:
        f = D / f"{pair}-15m-futures.feather"
        if f.exists():
            ohlcv[pair] = pd.read_feather(f).set_index("date")[["open","high","low","close","volume"]]
        ff = D / f"{pair}-1h-funding_rate.feather"
        if ff.exists():
            fr = pd.read_feather(ff).set_index("date")["open"]
            fr.index = fr.index.floor("15min")
            funding[pair] = fr[~fr.index.duplicated()]
    print(f"  {len(ohlcv)} 币 K线, {len(funding)} 币资金费")

    # 分 6 个月窗口运行
    all_trades = []
    all_equity = []
    capital = 150.0
    start_date = pd.Timestamp("2023-01-01", tz="UTC")
    end_date = pd.Timestamp("2026-08-27", tz="UTC")
    window = pd.DateOffset(months=6)

    current = start_date
    window_num = 0
    while current < end_date:
        w_end = min(current + window, end_date)
        window_num += 1
        print(f"\n窗口 {window_num}: {current.date()} → {w_end.date()} (起始资金 {capital:.2f}U)")

        trades, new_capital, equity = run_period(
            pred_df, ohlcv, funding,
            current, w_end, initial_capital=capital
        )

        w_ret = (new_capital / capital - 1) * 100 if capital > 0 else 0
        print(f"  交易: {len(trades)} | 收益: {w_ret:+.1f}% | 期末: {new_capital:.2f}U")

        all_trades.extend(trades)
        all_equity.extend(equity)
        capital = new_capital
        current = w_end

    # === 汇总 ===
    trades_df = pd.DataFrame(all_trades)
    equity_df = pd.DataFrame(all_equity)
    total_return = (capital / 150 - 1) * 100
    years = (end_date - start_date).days / 365.25
    cagr = ((capital / 150) ** (1/years) - 1) * 100

    # 最大回撤
    eq_series = equity_df.set_index("date")["equity"] if len(equity_df) > 0 else pd.Series([150])
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    max_dd = dd.min() * 100

    # Sharpe
    if len(equity_df) > 10:
        eq_daily = eq_series.resample("1D").last().dropna()
        daily_ret = eq_daily.pct_change().dropna()
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-12) * np.sqrt(365)
    else:
        sharpe = 0

    print(f"\n{'='*60}")
    print("ML v2 横截面 Top3/Bottom3 — 全期 NT 回测 (2023-01 → 2026-08)")
    print(f"{'='*60}")
    print(f"  初始资金: 150U → 最终: {capital:.2f}U")
    print(f"  总收益: {total_return:+.1f}%")
    print(f"  复合年化 (CAGR): {cagr:+.1f}%")
    print(f"  最大回撤: {max_dd:.1f}%")
    print(f"  Sharpe (日): {sharpe:.2f}")
    print(f"  总交易: {len(trades_df)}")

    if len(trades_df) > 0:
        print(f"  胜率: {(trades_df['ret']>0).mean()*100:.1f}%")
        print(f"  多头 PnL: {trades_df[trades_df['s']=='long']['pnl'].sum():+.1f}U")
        print(f"  空头 PnL: {trades_df[trades_df['s']=='short']['pnl'].sum():+.1f}U")

        # 分年
        trades_df["year"] = pd.to_datetime(trades_df["d"]).dt.year
        print(f"\n  分年:")
        for y, g in trades_df.groupby("year"):
            lg = g[g["s"]=="long"]; sg = g[g["s"]=="short"]
            yr_equity = equity_df[equity_df["date"].dt.year == y]["equity"]
            yr_start = yr_equity.iloc[0] if len(yr_equity) > 0 else 150
            yr_end = yr_equity.iloc[-1] if len(yr_equity) > 0 else 150
            yr_ret = (yr_end / yr_start - 1) * 100 if yr_start > 0 else 0
            print(f"    {y}: {len(g)}笔 多{lg['pnl'].sum():+.1f}U 空{sg['pnl'].sum():+.1f}U "
                  f"年收益 {yr_ret:+.1f}%")

    # 对比表
    print(f"\n{'='*60}")
    print("全项目策略对比（年化）")
    print(f"{'='*60}")
    print(f"  ML v2 横截面 (NT全期):  {cagr:+.1f}% ← 本回测")
    print(f"  MR bot (WFA):           +17%")
    print(f"  BTC急跌 (WFA):          +16%")
    print(f"  跨所套利 (dry-run):     +5-15%")
    print(f"  v1 ML 绝对收益:         ≈0%")

    # 保存
    trades_df.to_feather(ML_DIR / "nt_full_trades.feather")
    equity_df.to_feather(ML_DIR / "nt_full_equity.feather")
    (ML_DIR / "nt_full_summary.json").write_text(json.dumps({
        "total_return_pct": total_return, "cagr_pct": cagr,
        "max_drawdown_pct": max_dd, "sharpe": sharpe,
        "total_trades": len(trades_df),
        "final_capital": capital,
        "period": f"{start_date.date()} → {end_date.date()}",
    }, indent=2))
    print(f"\n耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    # 需要 PAIRS 变量
    CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    PAIRS = [p.split("/")[0] + "_USDT_USDT" for p in CONFIG["exchange"]["pair_whitelist"]]
    main()

# -*- coding: utf-8 -*-
"""跨所资金费率扫描器 + dry-run 模拟仓引擎（币安 vs OKX）

每轮（5分钟）:
  1. 拉币安前120流动性合约的实时费率+标记价格, 拉OKX全市场最新价(单次批量)
  2. 计算两所利差排名, 记录到 spreads_log.jsonl
  3. dry-run 账本（完整模拟）:
     - 开仓: 年化利差>=15% 且 24h 成交额>500万U, 双边各 NOTIONAL USDT 名义
     - 持仓: 按两所实时价格逐腿盯市(价格偏离损益), 按(当前费率/8h)比例计提资金费
     - 保证金: 每腿名义/3(模拟3x逐仓), 单腿浮亏>=85%保证金 => 模拟强平
     - 平仓: 利差收敛(<3%年化)或反向; 计提四腿手续费(各0.05%名义)
  4. --status 输出在持/历史盈亏分解(价格损益/资金费/手续费)
仅公开行情接口 + .env 里的密钥不需要(本脚本只用公开数据)。
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
SPREAD_LOG = HERE / "spreads_log.jsonl"
PAPER_POS = HERE / "paper_positions.json"
PAPER_CLOSED = HERE / "paper_trades.jsonl"

PROXY = "http://127.0.0.1:7890"
TOP_N = 120
# 调参(依据前5笔平仓教训: 小利差被手续费吃掉): 只做大鱼
OPEN_ANN = 0.50    # 年化利差>=50%才开(覆盖四腿费需约1天租金)
CLOSE_ANN = 0.05   # 收敛到5%以下才平
NOTIONAL = 30.0        # 每腿名义 USDT（对应双边各30U）
MAX_POS = 3            # 最多同时3组
LEV = 3.0              # 模拟杠杆(逐仓)
FEE_LEG = 0.0005       # 每腿 taker 0.05%（保守）
LIQ_LINE = 0.85        # 单腿浮亏达保证金85% => 模拟强平


def _bn_premium() -> dict:
    # URL 固定字面量, 域名 fapi.binance.com
    r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                     proxies={"http": PROXY, "https": PROXY}, timeout=15,
                     allow_redirects=False, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return {x["symbol"]: (float(x["lastFundingRate"]), float(x["markPrice"]))
            for x in r.json()}


def _bn_volumes() -> dict:
    r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                     proxies={"http": PROXY, "https": PROXY}, timeout=15,
                     allow_redirects=False, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return {x["symbol"]: float(x.get("quoteVolume", 0)) for x in r.json()}


def _okx_tickers() -> dict:
    # URL 固定字面量, 域名 www.okx.com; 一次拉全市场合约最新价
    r = requests.get("https://www.okx.com/api/v5/market/tickers",
                     params={"instType": "SWAP"},
                     proxies={"http": PROXY, "https": PROXY}, timeout=15,
                     allow_redirects=False, headers={"User-Agent": "Mozilla/5.0"})
    d = r.json()
    if d.get("code") != "0":
        return {}
    return {x["instId"]: float(x["last"]) for x in d["data"]}


def _okx_rate(inst_id: str) -> float | None:
    r = requests.get("https://www.okx.com/api/v5/public/funding-rate",
                     params={"instId": inst_id},
                     proxies={"http": PROXY, "https": PROXY}, timeout=15,
                     allow_redirects=False, headers={"User-Agent": "Mozilla/5.0"})
    d = r.json()
    if d.get("code") != "0" or not d.get("data"):
        return None
    return float(d["data"][0]["fundingRate"])


def fetch_all():
    prem = _bn_premium()
    # 成交额排名缓存 1 小时（ticker/24hr 权重 80, 高频拉会触发币安 -1003 封禁）
    vol_cache = HERE / ".vol_cache.json"
    vol = None
    if vol_cache.exists():
        import time as _t
        c = json.loads(vol_cache.read_text(encoding="utf-8"))
        if _t.time() - c["ts"] < 3600:
            vol = c["data"]
    if vol is None:
        vol = _bn_volumes()
        vol_cache.write_text(json.dumps({"ts": time.time(), "data": vol}), encoding="utf-8")
    okx_px = _okx_tickers()
    top = sorted(vol, key=vol.get, reverse=True)[:TOP_N]

    rows = []
    for sym in top:
        if not sym.endswith("USDT") or sym not in prem:
            continue
        base = sym[:-4]
        inst = f"{base}-USDT-SWAP"
        if inst not in okx_px:
            continue
        try:
            okx_rate = _okx_rate(inst)
        except Exception:
            continue
        if okx_rate is None:
            continue
        bn_rate, bn_px = prem[sym]
        spread = bn_rate - okx_rate
        rows.append({"base": base, "bn_rate": bn_rate, "okx_rate": okx_rate,
                     "spread": spread, "ann": spread * 3 * 365,
                     "bn_px": bn_px, "okx_px": okx_px[inst],
                     "vol_usd": vol[sym]})
        time.sleep(0.05)
    return rows


def leg_pnl(side: str, entry: float, cur: float, notional: float) -> float:
    """side: 'long'/'short'；返回盯市损益(USDT)。"""
    if side == "long":
        return (cur / entry - 1) * notional
    return (entry / cur - 1) * notional


def manage_paper(rows, now):
    by_base = {r["base"]: r for r in rows}
    pos = json.loads(PAPER_POS.read_text(encoding="utf-8")) if PAPER_POS.exists() else []
    kept = []

    for p in pos:
        r = by_base.get(p["base"])
        if r is None:  # 本轮未覆盖（流动性跌出前120），保持不动
            kept.append(p)
            continue
        # 1) 盯市价格损益（币安腿 + OKX腿）
        bn_leg = leg_pnl("short" if p["dir"] == "short_bn_long_okx" else "long",
                         p["entry_bn_px"], r["bn_px"], NOTIONAL)
        okx_leg = leg_pnl("long" if p["dir"] == "short_bn_long_okx" else "short",
                          p["entry_okx_px"], r["okx_px"], NOTIONAL)
        p["price_pnl"] = round(bn_leg + okx_leg, 4)
        # 2) 资金费按比例计提（当前利差 × 经过时间占8h比例）
        t_last = datetime.fromisoformat(p["last_funding_ts"])
        frac = min((now - t_last).total_seconds() / (8 * 3600), 1.0)
        p["funding_pnl"] = round(p["funding_pnl"] + p["spread_sign"] * r["spread"] * NOTIONAL * frac, 4)
        p["last_funding_ts"] = now.isoformat()
        p["last_ann"] = round(r["ann"], 4)
        # 3) 保证金模拟：单腿浮亏 vs 名义/LEV
        margin = NOTIONAL / LEV
        worst_leg = min(bn_leg, okx_leg)
        if worst_leg <= -LIQ_LINE * margin:
            close(p, now, "模拟强平(单腿浮亏触线)")
            continue
        # 4) 收敛/反向平仓
        if abs(r["ann"]) < CLOSE_ANN or (r["spread"] > 0) != (p["spread_sign"] > 0):
            close(p, now, f"利差收敛(年化{r['ann']*100:.0f}%)")
            continue
        kept.append(p)

    # 5) 开新仓
    held = {p["base"] for p in kept}
    for r in sorted(rows, key=lambda x: -abs(x["ann"])):
        if len(kept) >= MAX_POS:
            break
        if r["base"] in held or abs(r["ann"]) < OPEN_ANN or r["vol_usd"] < 5e6:
            continue
        kept.append({"base": r["base"], "open_ts": now.isoformat(),
                     "dir": "short_bn_long_okx" if r["spread"] > 0 else "long_bn_short_okx",
                     "spread_sign": 1 if r["spread"] > 0 else -1,
                     "open_ann": round(r["ann"], 4),
                     "entry_bn_px": r["bn_px"], "entry_okx_px": r["okx_px"],
                     "price_pnl": 0.0, "funding_pnl": 0.0,
                     "last_funding_ts": now.isoformat(), "last_ann": round(r["ann"], 4)})
        held.add(r["base"])
        print(f"[开仓] {r['base']} 利差年化{r['ann']*100:+.1f}% "
              f"{'币安空/OKX多' if r['spread'] > 0 else '币安多/OKX空'} 双边各{NOTIONAL:.0f}U")

    PAPER_POS.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    return kept


def close(p, now, reason):
    fees = 4 * FEE_LEG * NOTIONAL  # 开+平 双边共四腿
    net = p["price_pnl"] + p["funding_pnl"] - fees
    rec = {**p, "close_ts": now.isoformat(), "close_reason": reason,
           "fees": round(fees, 4), "net_pnl": round(net, 4)}
    with open(PAPER_CLOSED, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[平仓] {p['base']} {reason} | 价格{p['price_pnl']:+.2f} 资金费{p['funding_pnl']:+.2f} "
          f"费-{fees:.2f} => 净{net:+.2f}U")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)

    if args.status:
        pos = json.loads(PAPER_POS.read_text(encoding="utf-8")) if PAPER_POS.exists() else []
        print(f"dry-run 在持: {len(pos)} 组 (每腿 {NOTIONAL:.0f}U, {LEV:.0f}x)")
        for p in pos:
            net = p["price_pnl"] + p["funding_pnl"]
            print(f"  {p['base']:10s} {p['dir']:20s} 开仓年化{p.get('open_ann', p['last_ann'])*100:+6.1f}% "
                  f"现年化{p['last_ann']*100:+6.1f}% | 价格{p['price_pnl']:+6.2f} 资金费{p['funding_pnl']:+6.2f} 净{net:+6.2f}U")
        if PAPER_CLOSED.exists():
            closed = [json.loads(l) for l in PAPER_CLOSED.read_text(encoding="utf-8").splitlines()]
            tot = sum(c["net_pnl"] for c in closed)
            pp = sum(c["price_pnl"] for c in closed)
            fp = sum(c["funding_pnl"] for c in closed)
            print(f"已平仓: {len(closed)} 组 | 价格合计{pp:+.2f} 资金费合计{fp:+.2f} 净盈亏{tot:+.2f}U")
        return

    rows = fetch_all()
    rows.sort(key=lambda r: -abs(r["ann"]))
    with open(SPREAD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now.isoformat(), "top": rows[:15]}) + "\n")
    kept = manage_paper(rows, now)
    top = rows[0] if rows else None
    print(f"[{now.isoformat(timespec='seconds')}] 共同合约{len(rows)} 在持{len(kept)}" +
          (f" 最大利差 {top['base']} {top['ann']*100:+.1f}%/年" if top else ""))


if __name__ == "__main__":
    main()

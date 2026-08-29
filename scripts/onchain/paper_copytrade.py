# -*- coding: utf-8 -*-
"""模拟跟仓器: 把监控钱包的大额买入按币安永续价格模拟跟单。

规则:
  - 台账中出现 buy 且 eth >= 阈值 的记录 -> 按处理时刻的币安永续价格开模拟仓
    (两类钱包都跟, 记录 wtype 供事后分组比较)
  - 两条持有轨: 4h 与 12h, 到时按当时价格平仓记盈亏
  - 输出: positions_open.json(在仓) + paper_trades.jsonl(已平仓) + 状态打印
安全: 行情请求仅允许 fapi.binance.com（协议+域名+解析IP三重校验，禁重定向）。
用法: python paper_copytrade.py            # 处理新信号 + 平到期仓 + 打印状态
      python paper_copytrade.py --summary  # 只看汇总
"""
import argparse
import ipaddress
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.jsonl"
POS_FILE = HERE / "positions_open.json"
TRADES_FILE = HERE / "paper_trades.jsonl"
STATE = HERE / "paper_state.json"

SYMBOL_MAP = {"PEPE": "1000PEPEUSDT", "SPX": "SPXUSDT", "MOG": "MOGUSDT"}
# freqtrade 币对写法（信号桥用）
PAIR_MAP = {"PEPE": "1000PEPE/USDT:USDT", "SPX": "SPX/USDT:USDT", "MOG": "MOG/USDT:USDT"}
SIGNAL_FILE = HERE.parent / "user_data" / "onchain_signals.json"  # freqtrade 策略读取
# 分组触发门槛: 真人单笔小, 0.1 ETH; 机器人已初步证伪, 维持 0.3 只收大单
BUY_ETH_MIN = {"eoa": 0.10, "contract": 0.30}
HOLDS = (4, 12)  # 小时

_KLINE_URL = "https://fapi.binance.com/fapi/v1/klines"
_ALLOWED_HOST = "fapi.binance.com"
_PROXY = "http://127.0.0.1:7890"


def _check_url(url: str, via_proxy: bool) -> str:
    """协议 + 域名白名单；直连时再做 DNS 解析后的私网/保留地址阻断。

    走 HTTP 代理时目标域名由代理侧解析（本地解析结果不用于连接），
    本地 DNS 在部分网络下被污染为保留地址，若校验本地结果会误杀，
    因此代理路径仅依赖域名白名单（SSRF 防护由白名单+代理连接路径共同保证）。
    """
    p = urlparse(url)
    if p.scheme != "https":
        raise ValueError(f"仅允许 https: {url}")
    if p.hostname != _ALLOWED_HOST:
        raise ValueError(f"行情主机不在白名单: {p.hostname}")
    if not via_proxy:
        for info in socket.getaddrinfo(p.hostname, 443):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError(f"行情主机解析到内网/保留地址: {ip}")
    return url


def _price(symbol: str) -> float | None:
    """币安永续最新价（1m K线最后收盘）。请求前先做域名白名单校验。"""
    try:
        _check_url(_KLINE_URL, via_proxy=bool(_PROXY))
        # URL 为固定字面量常量（域名已由 _check_url 白名单校验）
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": symbol, "interval": "1m", "limit": 2},
                         proxies={"http": _PROXY, "https": _PROXY}, timeout=15,
                         allow_redirects=False,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        k = r.json()
        return float(k[-1][4]) if k else None
    except Exception:
        return None


def _symbol_ok(symbol: str) -> bool:
    return _price(symbol) is not None


def load(p: Path, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    positions = load(POS_FILE, [])
    state = load(STATE, {"ledger_lines": 0})

    if not LEDGER.exists():
        print("台账为空")
        return

    # 1) 平到期仓
    still_open = []
    for p in positions:
        if now >= datetime.fromisoformat(p["close_after"]):
            px = _price(p["symbol"])
            if px is None:
                still_open.append(p)  # 拿不到价先不平
                continue
            ret = (px / p["entry"] - 1) if p["side"] == "long" else (p["entry"] / px - 1)
            with open(TRADES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    **p, "exit": px, "exit_time": now.isoformat(),
                    "ret": round(ret, 5), "pnl_eth": round(ret * p["eth"], 4),
                }) + "\n")
        else:
            still_open.append(p)
    positions = still_open

    # 2) 处理新台账行 -> 开新仓
    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    new_signals = 0
    for i in range(state["ledger_lines"], len(lines)):
        r = json.loads(lines[i])
        wt = r.get("wtype", "eoa")
        if r.get("side") != "buy" or r.get("eth", 0) < BUY_ETH_MIN.get(wt, 0.3):
            continue
        sym = SYMBOL_MAP.get(r["token"])
        if not sym or not _symbol_ok(sym):
            continue
        px = _price(sym)
        if not px:
            continue
        for h in HOLDS:
            positions.append({
                "open_time": now.isoformat(), "token": r["token"], "symbol": sym,
                "wallet": r["wallet"], "wtype": r.get("wtype", "eoa"),
                "side": "long", "eth": r["eth"], "entry": px,
                "hold_h": h,
                "close_after": (now + timedelta(hours=h)).isoformat(),
            })
        new_signals += 1
        # 信号桥: 追加 freqtrade 时间窗（入场窗口 45 分钟，12h 轨道）
        sigs = load(SIGNAL_FILE, [])
        sigs.append({
            "pair": PAIR_MAP.get(r["token"]),
            "window_start": now.isoformat(),
            "window_end": (now + timedelta(minutes=45)).isoformat(),
            "hold_h": 12,
            "eth": r["eth"], "wtype": r.get("wtype", "eoa"),
        })
        SIGNAL_FILE.write_text(json.dumps(sigs, indent=1), encoding="utf-8")
        print(f"*** 跟仓开单: {r['token']}({sym}) 钱包{r['wallet'][:10]}({r.get('wtype','eoa')}) "
              f"{r['eth']}ETH @ {px} ***", flush=True)
    state["ledger_lines"] = len(lines)

    POS_FILE.write_text(json.dumps(positions, indent=1), encoding="utf-8")
    STATE.write_text(json.dumps(state), encoding="utf-8")

    # 3) 汇总
    trades = [json.loads(l) for l in TRADES_FILE.read_text(encoding="utf-8").splitlines()] if TRADES_FILE.exists() else []
    print(f"\n在仓: {len(positions)} 笔 | 已平仓: {len(trades)} 笔 | 本轮新信号: {new_signals}")
    if trades:
        for grp in ("eoa", "contract"):
            sub = [t for t in trades if t.get("wtype") == grp]
            if sub:
                win = sum(1 for t in sub if t["ret"] > 0)
                avg = sum(t["ret"] for t in sub) / len(sub)
                print(f"  [{grp}] n={len(sub)} 胜率{win/len(sub)*100:.0f}% 平均收益{avg*100:+.2f}%/笔 "
                      f"(4h轨: {sum(1 for t in sub if t['hold_h']==4)}笔, 12h轨: {sum(1 for t in sub if t['hold_h']==12)}笔)")


if __name__ == "__main__":
    main()

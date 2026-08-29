# -*- coding: utf-8 -*-
"""扫描币安 USDT-M 永续合约的上线日期，圈出新币研究宇宙。

用法（需代理环境变量）:
  HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 python scripts/scan_listings.py
输出: user_data/listings.json  [{symbol, base, onboard: ISO日期}, ...] 自 2023-01 起
"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import ccxt

ROOT = Path(__file__).resolve().parent.parent
SINCE = datetime(2023, 1, 1, tzinfo=timezone.utc)


def main():
    ex = ccxt.binance({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    markets = ex.load_markets()
    rows = []
    for sym, m in markets.items():
        if not m.get("swap") or m.get("quote") != "USDT" or not m.get("active"):
            continue
        if m.get("settle") != "USDT":
            continue
        ob = m.get("info", {}).get("onboardDate")
        if not ob:
            continue
        utype = m.get("info", {}).get("underlyingType", "COIN")
        if utype != "COIN":  # 剔除股票/指数等非加密标的永续
            continue
        dt = datetime.fromtimestamp(int(ob) / 1000, tz=timezone.utc)
        if dt < SINCE:
            continue
        base = m.get("base", "")
        rows.append({"symbol": sym, "base": base, "onboard": dt.date().isoformat()})

    rows.sort(key=lambda r: r["onboard"])
    out = ROOT / "user_data" / "listings.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"共 {len(rows)} 个 2023-01 之后上线的 USDT 永续")
    for r in rows[-8:]:
        print(" ", r["onboard"], r["base"])


if __name__ == "__main__":
    sys.exit(main())

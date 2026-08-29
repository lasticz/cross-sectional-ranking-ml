# -*- coding: utf-8 -*-
"""轻量下载新币 1h K线（freqtrade feather 格式，可断点续传）。

只拉价格不拉 funding/mark，速度远快于 freqtrade 下载器。
用法（需代理环境变量）:
  HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 python scripts/download_listings.py
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "user_data" / "data" / "binance" / "futures"


def main():
    listings = json.loads((ROOT / "user_data" / "listings.json").read_text(encoding="utf-8"))
    ex = ccxt.binance({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    ex.load_markets()

    done, fail = 0, []
    for item in listings:
        out = D / f"{item['base']}_USDT_USDT-1h-futures.feather"
        if out.exists():
            done += 1
            continue
        since = int(datetime.fromisoformat(item["onboard"]).replace(tzinfo=timezone.utc).timestamp() * 1000)
        rows = []
        try:
            while True:
                batch = ex.fetch_ohlcv(item["symbol"], timeframe="1h", since=since, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                since = batch[-1][0] + 1
                if len(batch) < 1000:
                    break
        except Exception as e:
            fail.append((item["base"], str(e)[:60]))
            time.sleep(2)
            continue
        if rows:
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
            df.to_feather(out)
            done += 1
        if done % 50 == 0:
            print(f"progress: {done}/{len(listings)}", flush=True)

    print(f"完成 {done}/{len(listings)}, 失败 {len(fail)}")
    for b, e in fail[:5]:
        print("  fail:", b, e)


if __name__ == "__main__":
    sys.exit(main())

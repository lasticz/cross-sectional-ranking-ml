# -*- coding: utf-8 -*-
"""补齐 binance USDT-M 合约资金费率历史。

自动读取 config.json 白名单，对比本地 funding 文件，只下载缺失的币。
输出格式与 freqtrade 下载的一致: OHLCV 形状，funding rate 存在 open 列，8h 一条。
绕过 fapi fundingRate 接口对默认 UA/密集请求的 403（浏览器 UA + 指数退避）。

用法: python scripts/fetch_funding.py [--proxy http://127.0.0.1:7890]
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "user_data" / "data" / "binance" / "futures"
START_MS = 1640995200000  # 2022-01-01 UTC
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def whitelist_bases() -> list[str]:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    bases = []
    for p in cfg["exchange"]["pair_whitelist"]:
        # "SOL/USDT:USDT" -> "SOL"
        bases.append(p.split("/")[0])
    return sorted(set(bases))


def fetch_pair(symbol: str, proxy: str | None) -> pd.DataFrame:
    rows, start = [], START_MS
    proxies = {"http": proxy, "https": proxy} if proxy else None
    while True:
        batch, wait = None, 30
        for _ in range(15):  # 403/超时等多为一过性，指数退避重试
            try:
                r = requests.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params={"symbol": symbol, "startTime": start, "limit": 1000},
                    headers={"User-Agent": UA},
                    proxies=proxies,
                    timeout=20,
                )
                if r.status_code == 200:
                    batch = r.json()
                    break
            except requests.RequestException:
                pass  # 网络瞬断，走退避重试
            time.sleep(wait)
            wait = min(wait * 2, 600)
        if batch is None:
            raise RuntimeError(f"{symbol}: 重试耗尽，仍被 403")
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(2.0)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df[["date"]].assign(open=df["fundingRate"].astype(float))
    for col in ("high", "low", "close", "volume"):
        df[col] = 0.0
    return df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    args = ap.parse_args()

    for base in whitelist_bases():
        out = DATA_DIR / f"{base}_USDT_USDT-1h-funding_rate.feather"
        if out.exists():
            continue
        df = fetch_pair(base + "USDT", args.proxy)
        df.to_feather(out)
        print(f"{base}: {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""交易所 API 密钥连通性验证（只读接口）。

密钥只从 ~/quant_8.25/.env 读取（KEY=VALUE 格式，chmod 600），本文件不含任何凭据。
验证: 币安合约账户余额读取 + OKX 账户余额读取（均走服务器本地代理）。
请求 URL 均为固定字面量（域名白名单: fapi.binance.com / www.okx.com），禁重定向。
用法: python3 verify_keys.py
"""
import base64
import hashlib
import hmac
import time
import urllib.parse
from pathlib import Path

import requests

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
PROXY = "http://127.0.0.1:7890"


def load_env() -> dict:
    env = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def check_binance(env) -> str:
    key, sec = env.get("BINANCE_API_KEY", ""), env.get("BINANCE_API_SECRET", "")
    if not key or not sec:
        return "缺少 BINANCE_API_KEY/SECRET"
    ts = int(time.time() * 1000)
    query = f"timestamp={ts}&recvWindow=10000"
    sig = hmac.new(sec.encode(), query.encode(), hashlib.sha256).hexdigest()
    # URL 为固定字面量, 域名 fapi.binance.com
    r = requests.get("https://fapi.binance.com/fapi/v2/balance",
                     params=query + "&signature=" + sig,
                     headers={"X-MBX-APIKEY": key},
                     proxies={"http": PROXY, "https": PROXY}, timeout=15,
                     allow_redirects=False)
    if r.status_code != 200:
        return f"失败 HTTP {r.status_code}: {r.text[:120]}"
    usdt = [x for x in r.json() if x.get("asset") == "USDT"]
    bal = usdt[0].get("balance") if usdt else "0"
    return f"OK, 币安合约 USDT 余额: {bal}"


def check_okx(env) -> str:
    key, sec, pp = env.get("OKX_API_KEY", ""), env.get("OKX_API_SECRET", ""), env.get("OKX_PASSPHRASE", "")
    if not key or not sec or not pp:
        return "缺少 OKX_API_KEY/SECRET/PASSPHRASE"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    msg = f"{ts}GET/api/v5/account/balance"
    sign = base64.b64encode(hmac.new(sec.encode(), msg.encode(), hashlib.sha256).digest()).decode()
    # ASCII 密码原样发送；仅非 ASCII（如全角！）才百分号编码（OKX SDK 同款处理）
    pp_enc = pp if pp.isascii() else urllib.parse.quote(pp, safe="")
    # URL 为固定字面量, 域名 www.okx.com
    r = requests.get("https://www.okx.com/api/v5/account/balance",
                     headers={"OK-ACCESS-KEY": key, "OK-ACCESS-SIGN": sign,
                              "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": pp_enc,
                              "Content-Type": "application/json"},
                     proxies={"http": PROXY, "https": PROXY}, timeout=15,
                     allow_redirects=False)
    d = r.json()
    if d.get("code") != "0":
        return f"失败: {str(d)[:150]}"
    details = d["data"][0].get("details", [])
    usdt = [x for x in details if x.get("ccy") == "USDT"]
    bal = usdt[0].get("availBal", "0") if usdt else "0"
    return f"OK, OKX 账户 USDT 可用: {bal}"


if __name__ == "__main__":
    if not ENV_FILE.exists():
        print(f"未找到 {ENV_FILE}，请先创建（见部署说明）")
        raise SystemExit(1)
    env = load_env()
    print("币安:", check_binance(env))
    print("OKX :", check_okx(env))

# -*- coding: utf-8 -*-
"""链上钱包盈亏扫描器（Phase 1: 钱包发现）

从 Uniswap V3 池的 Swap 事件重建近期交易，按钱包累计净买入量，
按当前价计算未实现盈亏的近似排名（赢家 = 在低位累计买入且仍在持有的钱包）。

安全: RPC 目标仅限白名单公共域名（协议+主机双重校验），代理仅允许 http(s)。
用法:
  HTTP_PROXY=... HTTPS_PROXY=... python scripts/onchain/wallet_pnl_scan.py --token 0x698... --days 5
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

# 只允许这些公共 RPC 域名（白名单，禁止内网/环回/私网目标）
# 注: publicnode 的 getLogs 需付费 token，drpc 免费开放
RPC_HOSTS = {
    "eth": "https://eth.drpc.org",
    "base": "https://base.drpc.org",
}

WETH = {
    "eth": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "base": "0x4200000000000000000000000000000000000006",
}

UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
GET_POOL_SELECTOR = "1698ee82"  # getPool(address,address,uint24) selector

SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

_ALLOWED_HOST_PREFIX = "eth.drpc.org", "base.drpc.org"


def _check_url(url: str) -> str:
    """协议 + 域名白名单 + DNS 解析后阻断私网/环回/链路本地 IP（防 SSRF 与 DNS rebinding）。"""
    import ipaddress
    import socket
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError(f"仅允许 http/https: {url}")
    if p.hostname not in _ALLOWED_HOST_PREFIX:
        raise ValueError(f"RPC 主机不在白名单: {p.hostname}")
    for info in socket.getaddrinfo(p.hostname, 443):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"RPC 主机解析到内网/保留地址，拒绝请求: {ip}")
    return url


def _check_proxy(proxy: str | None) -> str | None:
    if not proxy:
        return None
    p = urlparse(proxy)
    if p.scheme not in ("http", "https"):
        raise ValueError("代理仅允许 http/https")
    return proxy


def rpc(chain: str, method: str, params: list, proxy: str | None) -> dict:
    import requests
    url = _check_url(RPC_HOSTS[chain])
    px = _check_proxy(proxy)
    proxies = {"http": px, "https": px} if px else None
    body = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    r = requests.post(url, json=body, proxies=proxies, timeout=30, allow_redirects=False,
                      headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    r.raise_for_status()
    return r.json()


def get_pool(chain, token: str, proxy) -> str | None:
    """调用 Uniswap V3 factory.getPool(token, WETH, 3000)。"""
    token_a = token.lower().replace("0x", "").rjust(64, "0")
    token_b = WETH[chain].lower().replace("0x", "").rjust(64, "0")
    fee = (3000).to_bytes(32, "big").hex()
    data = "0x" + GET_POOL_SELECTOR + token_a + token_b + fee
    r = rpc(chain, "eth_call", [{"to": UNISWAP_V3_FACTORY, "data": data}, "latest"], proxy)
    result = r.get("result", "0x")
    if len(result) >= 66 and int(result, 16) != 0:
        return "0x" + result[-40:]
    return None


def fetch_swaps(chain, pool: str, from_block: int, to_block: int, proxy) -> list[dict]:
    """分块拉取 Swap 事件；出错时减半步长重试（有下限防死循环）。"""
    swaps = []
    cur = from_block
    step = 3000
    fails = 0
    while cur <= to_block:
        end = min(cur + step - 1, to_block)
        params = [{
            "address": pool,
            "topics": [SWAP_TOPIC],
            "fromBlock": hex(cur),
            "toBlock": hex(end),
        }]
        try:
            r = rpc(chain, "eth_getLogs", params, proxy)
            if "result" not in r:
                raise RuntimeError(str(r.get("error", r))[:80])
        except Exception as e:
            fails += 1
            if step > 200:
                step //= 2  # 结果太多/超时: 缩小区间
                print(f"  块 {cur}-{end} 失败({str(e)[:50]})，步长降为 {step}", flush=True)
                time.sleep(2)
                continue
            if fails > 60:
                print(f"  连续失败过多，跳到下一区间（该段数据缺失）", flush=True)
            else:
                time.sleep(3)
                continue
        else:
            fails = 0
            step = min(step * 2, 3000)
            swaps.extend(r["result"])
            if (end - from_block) % 30000 < step:
                print(f"  ...块 {cur}-{end}: 累计 {len(swaps)} 笔", flush=True)
        cur = end + 1
    return swaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True, help="代币合约地址")
    ap.add_argument("--chain", default="eth", choices=list(RPC_HOSTS))
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    args = ap.parse_args()

    latest = int(rpc(args.chain, "eth_blockNumber", [], args.proxy)["result"], 16)
    blocks_per_day = {"eth": 7200, "base": 43200}[args.chain]
    from_block = latest - args.days * blocks_per_day
    print(f"链: {args.chain} 最新块 {latest}, 扫描 {from_block} -> {latest} ({args.days}天)")

    pool = get_pool(args.chain, args.token, args.proxy)
    print(f"V3 0.3% 池: {pool}")
    if not pool:
        print("未找到池（检查代币地址/链）")
        return

    swaps = fetch_swaps(args.chain, pool, from_block, latest, args.proxy)
    print(f"Swap 事件总数: {len(swaps)}")

    buy_weth = defaultdict(float)
    sell_weth = defaultdict(float)
    token_amt = defaultdict(float)
    price_now = None

    for log in swaps:
        recipient = "0x" + log["topics"][2][-40:]
        data = log["data"][2:]
        amount0 = int.from_bytes(bytes.fromhex(data[0:64]), "big", signed=True)
        amount1 = int.from_bytes(bytes.fromhex(data[64:128]), "big", signed=True)
        tok, weth = abs(amount0), abs(amount1)
        if amount1 > 0:  # 池收到 token、付出 WETH -> 有人卖 token
            sell_weth[recipient] += weth / 1e18
            token_amt[recipient] -= tok
        else:            # 有人买 token
            buy_weth[recipient] += weth / 1e18
            token_amt[recipient] += tok
        if tok:
            price_now = weth / tok  # WETH wei / token raw

    rows = []
    for w, net_tok in token_amt.items():
        spent, got = buy_weth[w], sell_weth[w]
        if spent + got < 0.05:
            continue
        hold_eth = net_tok * (price_now or 0) / 1e18 if net_tok > 0 else 0.0
        rows.append({"wallet": w, "buy_eth": round(spent, 3), "sell_eth": round(got, 3),
                     "pnl_eth_approx": round((got - spent) + hold_eth, 3),
                     "net_token_raw": int(net_tok)})
    rows.sort(key=lambda r: -r["pnl_eth_approx"])
    out = Path(__file__).resolve().parent / "wallets_top.json"
    out.write_text(json.dumps(rows[:100], indent=1), encoding="utf-8")
    print(f"\n盈亏按当前价近似（ETH 计），Top 15 钱包:")
    for r in rows[:15]:
        print(f"  {r['wallet'][:10]}... 买{r['buy_eth']:>8.2f} 卖{r['sell_eth']:>8.2f} PnL≈{r['pnl_eth_approx']:+9.3f} ETH")
    print(f"\nTop100 已存 {out}")


if __name__ == "__main__":
    main()

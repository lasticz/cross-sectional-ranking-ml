# -*- coding: utf-8 -*-
"""每日钱包名单换血: 跨多个监控池重扫近 N 天, 排除合约钱包, 按多币合计 PnL 排名,
写回 monitor_config.json 的 wallets 字段（监控器每轮自动重读）。

过滤规则:
  - eth_getCode 非空 = 合约, 排除（防路由器/MEV bot）
  - 要求真实双向活动（买卖 ETH 合计 >= min_eth）
  - 排名 = 各币近似 PnL 合计
用法: python refresh_wallets.py [--days 5] [--top 10]
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from wallet_pnl_scan import SWAP_TOPIC, get_pool, rpc

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "monitor_config.json"
BLOCKS_PER_DAY = 7200
MIN_ETH = 0.3  # 5天总活动量门槛(从0.5下调: 扩大真人候选人池, 配合活跃度过滤)


def scan_pool(token: str, days: int, proxy: str):
    """返回 {wallet: pnl_eth_approx, buys, sells} 与该池 Swap 列表统计。"""
    pool = get_pool("eth", token, proxy)
    if not pool:
        return {}
    latest = int(rpc("eth", "eth_blockNumber", [], proxy)["result"], 16)
    from_block = latest - days * BLOCKS_PER_DAY
    # 只查近 N 天, 分块拉取
    swaps = []
    cur = from_block
    while cur <= latest:
        end = min(cur + 2000 - 1, latest)
        try:
            r = rpc("eth", "eth_getLogs", [{"address": pool, "topics": [SWAP_TOPIC],
                                            "fromBlock": hex(cur), "toBlock": hex(end)}], proxy)
            if "result" not in r:
                raise RuntimeError(str(r.get("error"))[:60])
            swaps.extend(r["result"])
        except Exception:
            end = cur + (end - cur) // 2 - 1 if end - cur > 200 else end
            if end <= cur:
                cur += 1
                continue
            continue
        cur = end + 1
    # token0 判断
    r0 = rpc("eth", "eth_call", [{"to": pool, "data": "0x0dfe1681"}, "latest"], proxy)
    tok0 = "0x" + r0.get("result", "0x")[-40:].lower()
    t0_is_tok = tok0 == token.lower()

    buy = defaultdict(float)
    sell = defaultdict(float)
    net_tok = defaultdict(float)
    last_blk = defaultdict(int)
    price = None
    for log in swaps:
        who = "0x" + log["topics"][2][-40:]
        d = log["data"][2:]
        a0 = int.from_bytes(bytes.fromhex(d[0:64]), "big", signed=True)
        a1 = int.from_bytes(bytes.fromhex(d[64:128]), "big", signed=True)
        tok, weth = (abs(a0), abs(a1)) if t0_is_tok else (abs(a1), abs(a0))
        flow = -a1 if t0_is_tok else a0  # >0 卖出所得, <0 买入花费
        if tok:
            price = weth / tok
        last_blk[who] = max(last_blk[who], int(log["blockNumber"], 16))
        if flow > 0:
            sell[who] += flow / 1e18
            net_tok[who] -= tok
        else:
            buy[who] += -flow / 1e18
            net_tok[who] += tok
    out = {}
    for w in set(buy) | set(sell):
        if buy[w] + sell[w] < MIN_ETH:
            continue
        hold_eth = net_tok[w] * (price or 0) / 1e18 if net_tok[w] > 0 else 0.0
        out[w] = {"pnl": sell[w] - buy[w] + hold_eth, "buy": buy[w], "sell": sell[w],
                  "last_blk": last_blk[w]}
    return out


def is_contract(wallet: str, proxy: str) -> bool:
    r = rpc("eth", "eth_getCode", [wallet, "latest"], proxy)
    return bool(r.get("result", "0x") not in ("0x", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    totals = defaultdict(lambda: {"pnl": 0.0, "tokens": 0, "last_blk": 0})
    latest = 0
    for sym, token in cfg["tokens"].items():
        res = scan_pool(token, args.days, args.proxy)
        print(f"[{sym}] {len(res)} 个活跃钱包", flush=True)
        for w, v in res.items():
            if v["pnl"] <= 0:
                continue
            totals[w]["pnl"] += v["pnl"]
            totals[w]["tokens"] += 1
            totals[w]["last_blk"] = max(totals[w]["last_blk"], v["last_blk"])
            latest = max(latest, v["last_blk"])

    # 活跃度门槛: 最近48h(≈14400块)内交易过才有资格(教训: 按盈亏选出的过路赢家不交易, 样本饿死)
    active_line = latest - 14400
    ranked = sorted(totals.items(), key=lambda kv: -kv[1]["pnl"])
    eoa, contract = [], []
    for w, info in ranked:
        if len(eoa) >= args.top and len(contract) >= args.top:
            break
        try:
            is_c = is_contract(w, args.proxy)
        except Exception:
            continue
        entry = {"address": w, "type": "contract" if is_c else "eoa"}
        if is_c and len(contract) < args.top:
            contract.append(entry)
            print(f"  入选[合约] {w[:12]}.. PnL≈{info['pnl']:+.2f} ETH")
        elif not is_c and len(eoa) < args.top:
            if info["last_blk"] < active_line:
                continue  # EOA 必须近48h活跃
            eoa.append(entry)
            print(f"  入选[EOA]   {w[:12]}.. PnL≈{info['pnl']:+.2f} ETH, {info['tokens']} 个币盈利, 近48h活跃")

    cfg["wallets"] = eoa + contract
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    CONFIG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"名单已更新: {len(eoa)} EOA + {len(contract)} 合约 -> monitor_config.json")


if __name__ == "__main__":
    main()

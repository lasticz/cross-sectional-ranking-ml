# -*- coding: utf-8 -*-
"""聪明钱钱包监控器（Phase 2）

对配置中的 top 钱包做池中心增量扫描: 每 N 分钟拉取监控池的新 Swap 事件,
匹配钱包名单记录台账; 钱包大额买入即时打印信号; --report 输出各钱包滚动盈亏。

用法:
  python wallet_monitor.py --once      # 扫一轮(补齐历史缺口)
  python wallet_monitor.py             # 常驻, 每 --interval 秒一轮
  python wallet_monitor.py --report    # 输出台账统计
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from wallet_pnl_scan import RPC_HOSTS, SWAP_TOPIC, _check_url, rpc

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "monitor_config.json"
STATE = HERE / "monitor_state.json"
LEDGER = HERE / "ledger.jsonl"

_TOKEN0_CACHE: dict[str, bool] = {}


def load_json(p: Path, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def get_pool(token: str, proxy) -> str | None:
    """V3 0.3% 池地址（复用扫描器里的 factory 调用，通过模块导入会带 UI 交互，这里独立实现）。"""
    from wallet_pnl_scan import UNISWAP_V3_FACTORY, WETH, GET_POOL_SELECTOR
    ta = token.lower().replace("0x", "").rjust(64, "0")
    tb = WETH["eth"].lower().replace("0x", "").rjust(64, "0")
    fee = (3000).to_bytes(32, "big").hex()
    data = "0x" + GET_POOL_SELECTOR + ta + tb + fee
    r = rpc("eth", "eth_call", [{"to": UNISWAP_V3_FACTORY, "data": data}, "latest"], proxy)
    res = r.get("result", "0x")
    return "0x" + res[-40:] if len(res) >= 66 and int(res, 16) != 0 else None


def token0_is_token(pool: str, token: str, proxy) -> bool:
    """池里 token0 是否为目标代币（决定 Swap 事件 amount0/amount1 的含义）。"""
    if pool in _TOKEN0_CACHE:
        return _TOKEN0_CACHE[pool]
    data = "0x0dfe1681"  # token0()
    r = rpc("eth", "eth_call", [{"to": pool, "data": data}, "latest"], proxy)
    res = r.get("result", "0x")
    tok0 = "0x" + res[-40:].lower() if len(res) >= 66 else ""
    flag = tok0 == token.lower()
    _TOKEN0_CACHE[pool] = flag
    return flag


def scan_once(cfg, proxy) -> int:
    state = load_json(STATE, {})
    latest = int(rpc("eth", "eth_blockNumber", [], proxy)["result"], 16)
    # wallets 兼容两种格式: 纯地址字符串 或 {"address","type"} 对象
    wtype = {}
    for w in cfg["wallets"]:
        if isinstance(w, dict):
            wtype[w["address"].lower()] = w.get("type", "eoa")
        else:
            wtype[w.lower()] = "eoa"
    new_rows = 0
    for sym, token in cfg["tokens"].items():
        time.sleep(2)  # 池间小间隔, 降低公共 RPC 限流压力
        pool = get_pool(token, proxy)
        if not pool:
            print(f"[{sym}] 未找到池，跳过")
            continue
        t0_is_tok = token0_is_token(pool, token, proxy)
        last = state.get(sym, latest - 300)  # 首轮只回看~1小时
        if last >= latest:
            continue
        logs = []
        cur = last + 1
        while cur <= latest:
            end = min(cur + 1500 - 1, latest)
            try:
                r = rpc("eth", "eth_getLogs", [{
                    "address": pool, "topics": [SWAP_TOPIC],
                    "fromBlock": hex(cur), "toBlock": hex(end)}], proxy)
                if "result" not in r:
                    raise RuntimeError(str(r.get("error"))[:60])
                logs.extend(r["result"])
            except Exception as e:
                msg = str(e)[:50]
                wait = 15 if "429" in msg else 2  # 限流用长退避
                print(f"[{sym}] getLogs {cur}-{end} 失败: {msg}", flush=True)
                time.sleep(wait)
                if end - cur > 200:
                    end = cur + (end - cur) // 2 - 1
                    continue
            cur = end + 1
        state[sym] = latest
        for log in logs:
            who = "0x" + log["topics"][2][-40:]
            if who.lower() not in wtype:
                continue
            data = log["data"][2:]
            a0 = int.from_bytes(bytes.fromhex(data[0:64]), "big", signed=True)
            a1 = int.from_bytes(bytes.fromhex(data[64:128]), "big", signed=True)
            tok_amt, weth = (abs(a0), abs(a1)) if t0_is_tok else (abs(a1), abs(a0))
            a1_is_weth_paid = (a1 > 0) if not t0_is_tok else (a0 < 0)
            # 统一含义: amount_weth>0 = 池付出WETH(有人卖token), <0 = 有人买token
            weth_flow = -a1 if t0_is_tok else a0  # >0 卖出所得, <0 买入花费
            blk = int(log["blockNumber"], 16)
            row = {"ts": datetime.now(timezone.utc).isoformat(), "block": blk,
                   "wallet": who, "wtype": wtype[who.lower()],
                   "token": sym, "side": "sell" if weth_flow > 0 else "buy",
                   "eth": round(abs(weth_flow) / 1e18, 4), "token_amt_raw": int(tok_amt)}
            with open(LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            new_rows += 1
            if row["side"] == "buy" and row["eth"] >= cfg.get("buy_signal_eth", 0.5):
                print(f"*** 信号 [{sym}] 钱包 {who[:10]}.. 买入 {row['eth']} ETH 等值 (块 {blk}) ***", flush=True)
    STATE.write_text(json.dumps(state), encoding="utf-8")
    return new_rows


def report():
    if not LEDGER.exists():
        print("台账为空")
        return
    from collections import defaultdict
    buy = defaultdict(float)
    sell = defaultdict(float)
    cnt = defaultdict(int)
    last_seen = {}
    wtypes = {}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        k = (r["wallet"], r["token"])
        (buy if r["side"] == "buy" else sell)[k] += r["eth"]
        cnt[k] += 1
        last_seen[r["wallet"]] = r["ts"]
        wtypes[r["wallet"]] = r.get("wtype", "eoa")
    print(f"{'钱包':14s} {'类型':5s} {'代币':6s} {'买ETH':>9s} {'卖ETH':>9s} {'笔数':>4s}  最近活动")
    for (w, t) in sorted(cnt, key=lambda k: -(sell[k] - buy[k])):
        print(f"{w[:12]}.. {wtypes[w]:5s} {t:6s} {buy[(w,t)]:9.2f} {sell[(w,t)]:9.2f} {cnt[(w,t)]:4d}  {last_seen[w][:19]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    args = ap.parse_args()

    if args.report:
        report()
        return
    cfg = load_json(CONFIG, {})
    _check_url(RPC_HOSTS["eth"])  # 启动即校验白名单
    if args.once:
        n = scan_once(cfg, args.proxy)
        print(f"本轮新增钱包交易 {n} 条")
        return
    print(f"常驻模式, 每 {args.interval}s 扫一轮, Ctrl+C 退出", flush=True)
    while True:
        try:
            cfg = load_json(CONFIG, cfg)  # 每轮重读配置, 支持外部程序换名单
            n = scan_once(cfg, args.proxy)
            print(f"[{datetime.now().isoformat(timespec='seconds')}] 新增 {n} 条", flush=True)
        except Exception as e:
            print(f"轮询异常: {str(e)[:80]}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

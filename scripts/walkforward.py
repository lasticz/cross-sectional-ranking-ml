# -*- coding: utf-8 -*-
"""Walk-forward 验证脚本

设计: 12 个月训练窗 + 3 个月测试窗，每 3 个月滚动一次（2023-01 → 2026-08 共 15 段）。
每段: 在训练窗对参数网格跑回测选最优（按总收益率），再用最优参数在测试窗跑一次。
拼接 15 段测试窗收益 = 复合的伪样本外业绩。

健壮性: 每次回测失败重试 1 次；训练全败时测试窗回退到先验默认参数；
进度增量写入 progress.json，重跑时跳过已完成的窗口（断点续跑）。

用法:
  python scripts/walkforward.py                    # 全部策略（自动续跑）
  python scripts/walkforward.py --strategies rotation mr
  python scripts/walkforward.py --max-windows 2    # 冒烟测试
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FT = r"C:\Users\18970\.conda\envs\quant\Scripts\freqtrade.exe"
OUT_DIR = ROOT / "user_data" / "walkforward_results"
PROGRESS = OUT_DIR / "progress.json"

TRAIN_MONTHS = 12
TEST_MONTHS = 3
STEP_MONTHS = 3
FIRST_TEST = (2023, 1)
LAST_TEST = (2026, 7)

# 只允许这些固定的策略名/环境变量/参数值（白名单，杜绝任意内容进入命令行）
PLAN = {
    "rotation": {
        "strategy": "RotationStrategy",
        "timeframe": "4h",
        "env": "ROT_LOOKBACK_DAYS",
        "grid": ("3", "5", "7", "10", "14"),
        "default": "7",
    },
    "mr": {
        "strategy": "MeanReversionStrategy",
        "timeframe": "15m",
        "env": "MR_RSI",
        "grid": ("25", "30", "35"),
        "default": "30",
    },
    "funding": {
        "strategy": "FundingExtremeStrategy",
        "timeframe": "1h",
        "env": "FUND_TH",
        "grid": ("-0.0003", "-0.0005", "-0.001"),
        "default": "-0.0005",
    },
    "shock": {
        "strategy": "BtcShockStrategy",
        "timeframe": "15m",
        "env": "BTC_SHOCK_TH",
        "grid": ("-0.003", "-0.005", "-0.008"),
        "default": "-0.005",
    },
}

RE_STRATEGY = re.compile(r"^[A-Za-z]\w{0,63}$")
RE_TIMEFRAME = re.compile(r"^\d+[mh]$")
RE_TIMERANGE = re.compile(r"^\d{8}-\d{8}$")
RE_PARAM = re.compile(r"^-?\d+(\.\d+)?$")


def add_months(ym, n):
    y, m = ym
    t = y * 12 + (m - 1) + n
    return t // 12, t % 12 + 1


def tr(ym0, ym1):
    return f"{ym0[0]}{ym0[1]:02d}01-{ym1[0]}{ym1[1]:02d}01"


def windows():
    out, cur = [], FIRST_TEST
    while cur <= LAST_TEST:
        out.append({
            "test_start": cur,
            "train": tr(add_months(cur, -TRAIN_MONTHS), cur),
            "test": tr(cur, add_months(cur, TEST_MONTHS)),
        })
        cur = add_months(cur, STEP_MONTHS)
    return out


def _bt_once(strategy, timeframe, timerange, env_kv, log_name):
    assert RE_STRATEGY.match(strategy), strategy
    assert RE_TIMEFRAME.match(timeframe), timeframe
    assert RE_TIMERANGE.match(timerange), timerange
    env = {**os.environ, **env_kv}
    cmd = [
        FT, "backtesting",
        "--userdir", "user_data",
        "--config", "config.json",
        "--strategy", strategy,
        "--timeframe", timeframe,
        "--timerange", timerange,
        "--cache", "none",
        "--export", "none",
    ]
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       timeout=1200, shell=False)
    text = p.stdout + p.stderr
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs" / log_name).write_text(text, encoding="utf-8", errors="replace")
    m = re.search(r"Total profit %\s*│\s*([+-]?[\d.]+)%", text)
    t = re.search(r"Total/Daily Avg Trades\s*│\s*(\d+) /", text)
    if not m:
        return {"profit": None, "trades": 0, "ok": False}
    return {"profit": float(m.group(1)), "trades": int(t.group(1)) if t else 0, "ok": True}


def run_bt(strategy, timeframe, timerange, env_kv, log_name):
    r = _bt_once(strategy, timeframe, timerange, env_kv, log_name)
    if not r["ok"]:  # 空输出/解析失败多为一过性，重试一次
        time.sleep(5)
        r = _bt_once(strategy, timeframe, timerange, env_kv, log_name)
    return r


def fmt_ym(ym):
    return f"{ym[0]}-{ym[1]:02d}"


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {}


def save_progress(progress):
    tmp = PROGRESS.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    tmp.replace(PROGRESS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", choices=list(PLAN), default=list(PLAN))
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    wins = windows()[: args.max_windows] if args.max_windows else windows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    for key in args.strategies:
        plan = PLAN[key]
        done = progress.get(key, {})
        print(f"\n===== {key} ({plan['strategy']} @ {plan['timeframe']}) =====", flush=True)
        for i, w in enumerate(wins):
            if str(i) in done:
                r = done[str(i)]
                print(f"  w{i:02d} [已缓存] {r['window']} param={r['param']}: {r['profit']}%", flush=True)
                continue
            best, best_profit = None, -1e9
            for v in plan["grid"]:
                assert RE_PARAM.match(v), v
                r = run_bt(plan["strategy"], plan["timeframe"], w["train"],
                           {plan["env"]: v}, f"{key}_w{i:02d}_train_{v}.log")
                if r["ok"] and r["profit"] > best_profit:
                    best, best_profit = v, r["profit"]
                print(f"  w{i:02d} train {fmt_ym(w['test_start'])} param={v}: {r['profit']}%", flush=True)
            fallback = best is None
            if fallback:
                best = plan["default"]
                print(f"  w{i:02d} 训练窗全败，回退默认参数 {best}", flush=True)
            tr_r = run_bt(plan["strategy"], plan["timeframe"], w["test"],
                          {plan["env"]: best}, f"{key}_w{i:02d}_test.log")
            row = {"window": fmt_ym(w["test_start"]), "param": best,
                   "fallback": fallback, **tr_r}
            done[str(i)] = row
            progress[key] = done
            save_progress(progress)  # 每窗落盘，断点续跑
            print(f"  w{i:02d} TEST {fmt_ym(w['test_start'])} param={best}: {tr_r['profit']}% ({tr_r['trades']} trades)", flush=True)

        comp = 1.0
        for r in done.values():
            if r["ok"]:
                comp *= 1 + r["profit"] / 100
        n_win = sum(1 for r in done.values() if r["ok"] and r["profit"] > 0)
        summary = {
            "strategy": key,
            "segments": list(done.values()),
            "oos_total_pct": round((comp - 1) * 100, 2),
            "win_segments": n_win,
            "total_segments": len(done),
        }
        (OUT_DIR / f"{key}_walkforward.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[{key}] OOS 复合收益: {summary['oos_total_pct']:+.2f}% | 盈利段: {n_win}/{len(done)}", flush=True)
        for r in done.values():
            flag = " (fallback)" if r.get("fallback") else ""
            print(f"  {r['window']}  param={r['param']:<8} {r['profit'] if r['ok'] else 'FAIL':>8}%  trades={r['trades']}{flag}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""ML v2 横截面 Top3/Bottom3 — NautilusTrader dry-run 实盘节点

架构 (与回测同一策略代码 XSTopBottomNT, 回测/实盘同源):
    行情: Binance Futures 实时 15m bar (真实行情, USDT-M)
    信号: 进程内 — 每个 8h 决策点 (00/08/16 UTC, bar 收盘桶触发) 用策略内
          滚动 K线缓冲 (~720 根/币) 走 research 同一条特征+模型代码
          (scripts/live/signals.py → 02_feature_families), 已通过
          validate_features.py 决策等价验证
    执行: SandboxExecutionClient (SimulatedExchange 撮合) — 真实行情 + 模拟成交,
          150 USDT 起始, 3x 杠杆 MARGIN 账户, reduce-only 止损单生效
          → 不产生真实订单, 无需交易权限

与回测的时序对齐:
    回测: 决策 bar (ts=T) 桶满触发, 市价单按 T 收盘成交 ≈ 07 引擎的下一根开盘
    实盘: T+15:00 后 27 币 bar 全部到齐 → 桶满触发 → 特征仅用 ≤T 的 bar →
          市价单按最新 bar (T) 收盘附近成交。与回测同一成交时点语义。

已知边界 (dry-run 阶段接受):
    - 资金费率未入账 (回测 NT 口径同样不含; 每日报告另行核对)
    - 重启后仓位丢失: 净值通过 state.json 续接 (余额+未实现), 仓位清零
    - 决策点快照 <10 币 / 模型计算异常 → 跳过该次调仓 (持仓+止损照常)

状态输出 (供日常检查/日报):
    {status_dir}/status.json   最近决策/净值/持仓/计数器 (每决策点+10min 定时)
    {status_dir}/trades.jsonl  逐笔平仓记录 (追加)
    {status_dir}/state.json    净值续接文件 (重启时读入为 sandbox 起始余额)

用法:
    C:/Users/18970/.conda/envs/quant/python.exe scripts/live/nt_dryrun_node.py
    ... --proxy http://127.0.0.1:7890       # 服务器 mihomo
    ... --balance 150                       # 首次起始净值 (有 state.json 时忽略)
    ... --smoke                             # 每 15m 决策一次 (验证下单链路, 勿长跑)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "engine"))
sys.path.insert(0, str(ROOT / "scripts" / "live"))

from nt_xs_ml_backtest import TIMEFRAME_STR, VENUE, XSTopBottomNT, ts_ns  # noqa: E402
import signals as SIG  # noqa: E402

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType  # noqa: E402
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig  # noqa: E402
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory  # noqa: E402
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig  # noqa: E402
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory  # noqa: E402
from nautilus_trader.common.config import InstrumentProviderConfig  # noqa: E402
from nautilus_trader.config import (
    LiveDataEngineConfig,
    LiveRiskEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)  # noqa: E402
from nautilus_trader.live.node import TradingNode  # noqa: E402
from nautilus_trader.model.data import BarType  # noqa: E402
from nautilus_trader.model.identifiers import InstrumentId, TraderId  # noqa: E402

NS_8H = 8 * 3600 * 1_000_000_000
NS_BAR_NS = 15 * 60 * 1_000_000_000   # 15m bar 长度; close_time = open + 15m - 1ms
BTC_BASE = "BTC_USDT_USDT"
BTC_SYMBOL = "BTCUSDT-PERP"
HIST_CAP = 1500          # 每币缓冲 bar 上限 (~15 天)
WARMUP_MIN_BARS = 390    # 决策前单币最少 bar 数 (最深回看链 384 + 余量)


class XSTopBottomLive(XSTopBottomNT):
    """进程内信号版 XSTopBottomNT — 决策由实时 K线缓冲 + 部署模型在线计算"""

    def __init__(
        self,
        bases: list[str],
        ml_dir: Path,
        config=None,
        status_dir: Path | None = None,
        proxy: str | None = None,
        smoke: bool = False,
        min_snap: int = SIG.MIN_SNAP_COINS,
    ):
        super().__init__(
            decisions={},
            base2sym={},          # on_start 从 cache 填充
            instruments={},       # on_start 从 cache 填充
            config=config,
        )
        self._bases = bases
        self._ml_dir = ml_dir
        self._status_dir = status_dir or (ROOT / "user_data" / "ml_v2" / "live")
        self._proxy = proxy
        self._smoke = smoke
        self._min_snap = min_snap

        self._hist: dict[str, dict[int, tuple]] = {}   # base -> {ts_ns: (o,h,l,c,v)}
        self._btc_hist: dict[int, tuple] = {}
        self._btc_sym = BTC_SYMBOL
        self._reg = self._clf = None
        self._feat_cols = self._med = None
        self._n_skipped = 0
        self._last_decision_iso = None
        self._status_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def on_start(self):
        self._load_models()
        self._populate_instruments()
        super().on_start()  # 订阅 27 币 bar
        self._subscribe_btc()
        self._seed_history()
        self._write_status("start")
        self.clock.set_timer("status", timedelta(minutes=10), callback=self._on_timer)

    def _load_models(self):
        self._reg, self._clf, self._feat_cols, self._med, meta = SIG.load_models(self._ml_dir)
        self.log.info(
            f"模型加载: reg+clf (训练至 {meta.get('train_through')}), "
            f"{len(self._feat_cols)} 特征, 空头门槛 {meta.get('short_conf_threshold')}"
        )

    def _populate_instruments(self):
        """从 cache 取 instrument (数据客户端 connect 时已加载); 缺失的币种剔除"""
        deadline = time.time() + 60
        while time.time() < deadline:
            insts = self.cache.instruments(venue=VENUE)
            if len(insts) >= len(self._bases):
                break
            self.log.warning(f"等待 instrument 加载: cache={len(insts)}/{len(self._bases)}")
            time.sleep(2)
        insts = {i.id.symbol.value: i for i in self.cache.instruments(venue=VENUE)}
        missing = []
        for base in self._bases:
            sym = base.split("_")[0] + "USDT-PERP"
            inst = insts.get(sym)
            if inst is None:
                missing.append(sym)
                continue
            self._base2sym[base] = sym
            self._instruments[sym] = inst
        self._sym2base = {v: k for k, v in self._base2sym.items()}
        self._n_sub = len(self._instruments)
        if missing:
            self.log.error(f"缺失 instrument (跳过交易): {missing}")
        self.log.info(f"trading universe: {self._n_sub} 币")

    def _subscribe_btc(self):
        if self._btc_sym in {i.id.symbol.value for i in self.cache.instruments(venue=VENUE)}:
            bt = BarType.from_str(f"{self._btc_sym}.{VENUE}-{TIMEFRAME_STR}-LAST-EXTERNAL")
            self.subscribe_bars(bt)

    def _seed_history(self):
        """REST 预填 K线缓冲 (否则自然积累 720 根要 7.5 天)"""
        self.log.info("REST 预填历史K线 ...")
        try:
            kl, btc = SIG.fetch_all(self._bases, proxy=self._proxy)
        except Exception as e:  # noqa: BLE001
            self.log.error(f"历史K线预填失败 (将自然积累, 数日内无决策): {e}")
            return
        for base, df in kl.items():
            self._hist[base] = {
                ts_ns(pd.Timestamp(d)): (r.open, r.high, r.low, r.close, r.volume)
                for d, r in zip(df["date"], df.itertuples(index=False))
            }
        self._btc_hist = {
            ts_ns(pd.Timestamp(d)): (r.open, r.high, r.low, r.close, r.volume)
            for d, r in zip(btc["date"], btc.itertuples(index=False))
        }
        sizes = sorted(len(v) for v in self._hist.values())
        self.log.info(f"预填完成: min={sizes[0]} max={sizes[-1]} bars/币, BTC={len(self._btc_hist)}")

    def _on_timer(self, event):
        self._write_status("timer")

    # ------------------------------------------------------------------ #
    # bar 流
    # ------------------------------------------------------------------ #
    def on_bar(self, bar):
        sym = bar.bar_type.instrument_id.symbol.value
        base = self._sym2base.get(sym, BTC_BASE if sym == self._btc_sym else None)
        if base is not None:
            # 实盘 bar.ts_event = kline close_time (open + 15m - 1ms), 而回测/
            # feather 时间轴 = open_time → 统一换算成 open_time 存缓冲, 保证与
            # 离线特征管线同一时间轴 (REST 预填数据也是 open_time)
            key = bar.ts_event + 1_000_000 - NS_BAR_NS
            h = self._btc_hist if base == BTC_BASE else self._hist.setdefault(base, {})
            h[key] = (
                bar.open.as_double(), bar.high.as_double(), bar.low.as_double(),
                bar.close.as_double(), bar.volume.as_double(),
            )
            if len(h) > HIST_CAP:
                for k in sorted(h.keys())[: len(h) - HIST_CAP]:
                    del h[k]
        super().on_bar(bar)

    # ------------------------------------------------------------------ #
    # 决策 (替换父类的静态 decisions 查表)
    # ------------------------------------------------------------------ #
    def _maybe_decide(self, ts: int, present: set, px_snap: dict):
        """ts 为父类桶键 = 实盘 bar.ts_event = close_time。
        决策网格: 收盘于 00/08/16 UTC 的那根 bar → (ts + 1ms) % 8h == 0。
        """
        if ts in self._decided:
            return
        if not self._smoke and (ts + 1_000_000) % NS_8H != 0:
            return
        if ts not in self._decisions:
            sig = self._compute_decision(ts + 1_000_000 - NS_BAR_NS)  # open_time 口径
            if sig is None:
                self._n_skipped += 1
                self._write_status("skip")
                return
            self._decisions[ts] = sig
        self._decide(ts, present, px_snap)
        self._last_decision_iso = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).isoformat()
        self._write_status("decide")

    def _hist_to_df(self, hist: dict, upto_ts: int) -> pd.DataFrame | None:
        rows = [((k // 1_000_000_000), *v) for k, v in hist.items() if k <= upto_ts]
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["s", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["s"], unit="s", utc=True)
        return df.drop(columns=["s"])

    def _compute_decision(self, ts: int) -> tuple | None:
        """K线缓冲 → 特征 → 预测 → (ranked, conf_dn)。任何异常 → None (跳过调仓)"""
        try:
            klines = {}
            for base, h in self._hist.items():
                df = self._hist_to_df(h, ts)
                if df is not None and len(df) >= 32:
                    klines[base] = df
            btc_df = self._hist_to_df(self._btc_hist, ts)
            if btc_df is None or len(btc_df) < WARMUP_MIN_BARS:
                # BTC 流数据不足 → REST 兜底
                btc_df = SIG.fetch_klines("BTCUSDT", proxy=self._proxy)
            short_bases = [b for b, df in klines.items() if len(df) < WARMUP_MIN_BARS]
            if short_bases:
                self.log.warning(f"历史不足 {WARMUP_MIN_BARS} 根 (特征将部分为NaN): {short_bases}")

            snap = SIG.compute_signal(
                klines, btc_df, self._reg, self._clf, self._feat_cols, self._med,
                min_snap=self._min_snap,
            )
            if snap is None:
                self.log.error(f"决策跳过 @ {ts}: 快照币数不足")
                return None
            self._last_snap = snap
            top = snap["ranked"][:3]
            bot = snap["ranked"][-3:]
            self.log.info(
                f"决策 @ {snap['date']}: coins={snap['n_coins']} "
                f"LONG={top} SHORT候选={[(b, round(snap['conf_dn'][b], 3)) for b in bot]}"
            )
            return (tuple(snap["ranked"]), snap["conf_dn"])
        except Exception as e:  # noqa: BLE001 — 信号计算故障只跳过本次调仓
            self.log.error(f"决策计算异常 @ {ts}: {e!r}")
            return None

    # ------------------------------------------------------------------ #
    # 状态落盘
    # ------------------------------------------------------------------ #
    def on_position_closed(self, event):
        super().on_position_closed(event)
        t = self._closed_trades[-1] if self._closed_trades else None
        if t is not None:
            with open(self._status_dir / "trades.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({**t, "_iso": datetime.now(timezone.utc).isoformat()},
                                   ensure_ascii=False) + "\n")

    def _positions_snapshot(self) -> list[dict]:
        return [
            {
                "symbol": pos.instrument_id.symbol.value,
                "side": pos.side.name,
                "qty": pos.quantity.as_double(),
                "avg_px": pos.avg_px_open.as_double(),
            }
            for pos in self.cache.positions_open(venue=VENUE)
        ]

    def _write_status(self, trigger: str):
        try:
            snap = getattr(self, "_last_snap", None)
            payload = {
                "trigger": trigger,
                "iso": datetime.now(timezone.utc).isoformat(),
                "equity": round(self._equity(), 2),
                "n_positions": len(self.cache.positions_open(venue=VENUE)),
                "positions": self._positions_snapshot(),
                "n_decisions": len(self._decided),
                "n_skipped": self._n_skipped,
                "n_entries": self._n_entries,
                "n_closed": len(self._closed_trades),
                "n_rejected_entry": self._n_rejected_entry,
                "last_decision": self._last_decision_iso,
                "last_top3": snap["ranked"][:3] if snap else None,
                "last_bottom3": snap["ranked"][-3:] if snap else None,
            }
            tmp = self._status_dir / "status.json.tmp"
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self._status_dir / "status.json")
            state = {"equity": payload["equity"],
                     "iso": payload["iso"]}
            tmp2 = self._status_dir / "state.json.tmp"
            tmp2.write_text(json.dumps(state), encoding="utf-8")
            tmp2.replace(self._status_dir / "state.json")
        except Exception as e:  # noqa: BLE001 — 状态写盘失败不影响交易
            self.log.warning(f"status 写盘失败: {e}")

    def on_stop(self):
        self._write_status("stop")
        super().on_stop()


# --------------------------------------------------------------------------- #
# 节点装配
# --------------------------------------------------------------------------- #
def _preload_instruments(load_ids: list[InstrumentId], proxy: str | None) -> int:
    """REST 预加载 instrument 到共享 cache。

    必须在 node.build()/start 之前完成: sandbox exec 的 connect() 从 cache
    读取该 venue 的 instrument 挂入 SimulatedExchange, 而它连接早于 Binance
    数据客户端就绪 (实测 cache=0) → 不预加载则全部订单 'no market' 被拒。
    """
    import asyncio

    from nautilus_trader.adapters.binance.futures.providers import (
        BinanceFuturesInstrumentProvider,
    )
    from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
    from nautilus_trader.common.component import LiveClock

    async def run() -> list:
        client = BinanceHttpClient(
            clock=LiveClock(),
            api_key=None,
            api_secret=None,
            base_url="https://fapi.binance.com",
            proxy_url=proxy,
        )
        provider = BinanceFuturesInstrumentProvider(
            client=client,
            clock=LiveClock(),
            account_type=BinanceAccountType.USDT_FUTURES,
            config=InstrumentProviderConfig(load_all=False, load_ids=tuple(load_ids)),
            venue=VENUE,
        )
        await provider.load_ids_async(list(load_ids))
        return list(provider.get_all().values())

    return asyncio.run(run())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proxy", default=os.environ.get("NT_PROXY"),
                    help="HTTP 代理 (服务器: http://host.docker.internal:7890)")
    ap.add_argument("--balance", type=float, default=150.0, help="首次起始净值 USDT")
    ap.add_argument("--status-dir", type=Path, default=ROOT / "user_data" / "ml_v2" / "live")
    ap.add_argument("--ml-dir", type=Path, default=ROOT / "user_data" / "ml_v2")
    ap.add_argument("--smoke", action="store_true", help="每 15m 决策一次 (链路验证)")
    args = ap.parse_args()

    bases = SIG.whitelist_bases()
    coins = [b.split("_")[0] for b in bases]
    load_ids = [InstrumentId.from_str(f"{c}USDT-PERP.BINANCE") for c in coins + ["BTC"]]

    # 净值续接: 重启后以 state.json 的期末净值作为 sandbox 起始余额 (仓位清零)
    balance = args.balance
    state_path = args.status_dir / "state.json"
    if state_path.exists():
        try:
            balance = json.loads(state_path.read_text(encoding="utf-8"))["equity"]
            print(f"净值续接: {balance:.2f} USDT (from {state_path})")
        except Exception:  # noqa: BLE001
            print("state.json 读取失败, 使用默认起始净值")

    data_cfg = BinanceDataClientConfig(
        account_type=BinanceAccountType.USDT_FUTURES,
        api_key=os.environ.get("BINANCE_API_KEY"),
        api_secret=os.environ.get("BINANCE_API_SECRET"),
        proxy_url=args.proxy,
        instrument_provider=InstrumentProviderConfig(load_all=False, load_ids=tuple(load_ids)),
        update_instruments_interval_mins=60,
    )
    exec_cfg = SandboxExecutionClientConfig(
        venue=str(VENUE),
        starting_balances=[f"{balance:.2f} USDT"],
        oms_type="NETTING",
        account_type="MARGIN",
        default_leverage=Decimal(3),
        bar_execution=True,
        use_reduce_only=True,
    )
    node_cfg = TradingNodeConfig(
        trader_id=TraderId("XSML-001"),
        logging=LoggingConfig(log_level="INFO"),
        data_engine=LiveDataEngineConfig(),
        # 实测: live RiskEngine 对市价单做名义价值校验时因无 quote 而以
        # 'no market for X' 拒单 (未到 sandbox 撮合层)。dry-run 由策略自身
        # 逻辑 + 撮合引擎把关, 旁路风控校验。
        risk_engine=LiveRiskEngineConfig(bypass=True),
        data_clients={"BINANCE": data_cfg},
        exec_clients={"SANDBOX": exec_cfg},
    )
    node = TradingNode(config=node_cfg)
    # NT 1.231: 客户端工厂必须显式注册, 否则 build 时 "No factory registered"
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("SANDBOX", SandboxLiveExecClientFactory)

    insts = _preload_instruments(load_ids, args.proxy)
    for inst in insts:
        node.cache.add_instrument(inst)
    print(f"预加载 instrument: {len(insts)} 个 → cache")
    node.build()

    # NT 1.231 sandbox 适配器缺陷: connect 只订阅 data.*.BINANCE.* (匹配
    # instrument 主题), 而 bar 主题为 data.bars.{symbol}.{venue}-{spec}
    # (venue 在第 4 段) → 撮合账本永远收不到行情, 市价单全被 'no market' 拒。
    # 此处给 sandbox 按精确主题补订阅全部 bar 流。
    sandbox = next(
        c for c in node.kernel.exec_engine._clients.values()
        if type(c).__name__ == "SandboxExecutionClient"
    )
    n_sub = 0
    for c in coins + ["BTC"]:
        bt = BarType.from_str(f"{c}USDT-PERP.{VENUE}-15-MINUTE-LAST-EXTERNAL")
        sandbox._msgbus.subscribe(f"data.bars.{bt}", handler=sandbox.on_data)
        n_sub += 1
    print(f"sandbox 补订阅 bar 流: {n_sub} 个")

    strategy = XSTopBottomLive(
        bases=bases,
        ml_dir=args.ml_dir,
        status_dir=args.status_dir,
        proxy=args.proxy,
        smoke=args.smoke,
    )
    node.trader.add_strategy(strategy)

    print(f"dry-run 节点启动: {len(bases)} 币, 起始 {balance:.2f} USDT, "
          f"proxy={args.proxy or '无'}, smoke={args.smoke}")
    try:
        node.run()
    except KeyboardInterrupt:
        print("收到 Ctrl+C, 停止节点 ...")
    finally:
        node.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())

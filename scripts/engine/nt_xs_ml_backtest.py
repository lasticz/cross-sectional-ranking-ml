# -*- coding: utf-8 -*-
"""NautilusTrader 1.231.0 — ML v2 横截面 Top3/Bottom3 策略回测（wf_preds_v2 移植）

信号源: user_data/ml_v2/wf_preds_v2.feather (idx, date, base, pred, conf_up, conf_dn)
参考实现: scripts/ml_v2/07_full_ls_backtest.py (pandas 引擎) + scripts/engine/nt_mr_backtest.py (NT 基础设施)

策略规则（任务规约）:
  决策点:   每 32 根 15m bar (8h), 取预测时间戳网格 dates[::32] → 每天 00:00/08:00/16:00 UTC
  排名:     按 pred 降序; 多头 = Top3 (无门槛), 空头 = Bottom3 (仅 conf_dn > 0.50)
  缓冲区:   多头在 Top5 内不换出, 空头在 Bottom5 内不换出; 跌出缓冲区离场
  最小持有: 32 根 bar (8h) — 决策节奏 8h 一次, 结构性满足 (实现中仍显式检查)
  止损:     入场价反向 15% (reduce-only stop-market 托管, 任一 bar 的 high/low 可触发)
  仓位:     净值(含未实现盈亏) 10% 保证金 × 3x 杠杆 = 名义净值 30%/仓, 最多 3 多 + 3 空
  费用:     maker = taker = 0.02% (与 07 引擎 / config.json 一致)

成交时机（复用 nt_mr_backtest.py 已验证的模式）:
  决策在 "该时间戳全部应有 bar 到齐" (桶满, alive_count) 或 "下一时间戳首根 bar 到达"
  (桶边界) 时触发, 市价单立即按决策 bar 的 close 成交 ≈ 参考引擎的下一根 bar 开盘价
  (nopen, 实测逐笔一致)。同一时间戳内 bar 顺序随机化(固定种子), 使边界触发时触发币
  订单簿领先一根 bar 的微小偏差在各币种间均匀分布。
  期末(最后一个决策点)强制平掉全部仓位 → 总收益为纯已实现口径。

实现要点（NT 1.231.0 异步事件模型）:
  NT 回测中订单按提交时点的订单簿成交(价格正确), 但 OrderFilled/Position* 事件、
  cache 与 account 状态经消息队列异步送达 → on_position_* 驱动的状态在同一决策内
  是滞后的。本策略维护同步影子账本 _live/_meta/_closing(提交订单时立即记账),
  事件回调仅用于交易记录与对账, 从而支持与参考引擎一致的"同一决策内先平后开/反手"。

验证（2023-01-01 ~ 2023-06-01, 24 币, 454 决策点）:
  - 与 07 pandas 引擎逐决策持仓状态对照 (Jan 1-5): 完全一致
  - --short-buffer 3 (对齐 07 的空头离场): NT +111.32% / 1704 笔 vs
    pandas 同规则 +104.89% / 1707 笔 — 差异来自止损微观结构
    (NT 连续 stop-market 触发价 vs pandas 决策 bar max(low, level))
  - 07 原版(决策bar止损+Bottom3空头离场, 不强平): +122.62% / 1112 笔

已知差异（相对 07 pandas 引擎）:
  - 07 的空头离场是跌出 Bottom3 即走; 本移植按任务规约默认 Bottom5 缓冲 (--short-buffer 可调)
  - 07 只在决策 bar (每 8h) 检查止损; NT 托管 stop-market 每 15m bar 都生效
  - 07 期末不强平(未实现盈亏不计入); NT 期末强平, 收益为完整已实现口径
  - NT 不含资金费率 (对照时给 07 传 funding={} 同样关闭, 保证口径一致)
  - 决策内平仓的已实现盈亏在 NT 中异步结算, 该决策新开仓的 sizing 净值略滞后(幅度 <0.6%)

结果（默认参数, 初始 150 USDT）:
  +88.21% | 1605 笔 (736 多 / 869 空) | 胜率 52.8% | 最大回撤 -24.33%
  分月: 2023-01 -6.74% / 02 +43.73% / 03 +14.59% / 04 +33.30% / 05 -8.00%

用法:
  python scripts/engine/nt_xs_ml_backtest.py                        # 27 币(窗口内 24 币有效), 2023-01-01 ~ 2023-06-01
  python scripts/engine/nt_xs_ml_backtest.py --pairs 10             # 前 10 币快速验证
  python scripts/engine/nt_xs_ml_backtest.py --compare              # 结束后跑 07 pandas 引擎同窗口对照
  python scripts/engine/nt_xs_ml_backtest.py --short-buffer 3       # 复现 07 的空头离场规则
  XS_DEBUG=1 ...                                                    # 打印逐决策持仓 trace
"""
from __future__ import annotations
from __future__ import annotations

import argparse
import decimal
import importlib.util
import itertools
import json
import os
import random
import time
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import (
    AccountType,
    CurrencyType,
    OmsType,
    OrderSide,
    OrderType,
    PositionSide,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy as NTStrategy

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "user_data" / "data" / "binance" / "futures"
PRED_FILE = ROOT / "user_data" / "ml_v2" / "wf_preds_v2.feather"
CONFIG_PATH = ROOT / "config.json"
REF_ENGINE = ROOT / "scripts" / "ml_v2" / "07_full_ls_backtest.py"

VENUE = Venue("BINANCE")
TIMEFRAME_STR = "15-MINUTE"
BAR_SEC = 15 * 60
NS_BAR = BAR_SEC * 1_000_000_000


# ---------------------------------------------------------------------------
# 工具（复用 nt_mr_backtest.py）
# ---------------------------------------------------------------------------
def ts_ns(dt: pd.Timestamp) -> int:
    """UTC 时间戳 → 纳秒（与 Bar.ts_event 同一公式，保证 lookup key 一致）"""
    return int(dt.timestamp() * 1e9)


def load_pairs(pairs_limit: int | None) -> list[tuple[str, str]]:
    """返回 [(preds_base, symbol), ...] 例如 ("ETH_USDT_USDT", "ETHUSDT-PERP")"""
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    out = []
    for p in cfg["exchange"]["pair_whitelist"]:
        base = p.split("/")[0]
        prefix = f"{base}_USDT_USDT"
        out.append((prefix, f"{base}USDT-PERP"))
    if pairs_limit:
        out = out[:pairs_limit]
    return out


def get_or_create_currency(code: str) -> Currency:
    import nautilus_trader.model.currencies as predefined

    existing = getattr(predefined, code, None)
    if existing is not None:
        return existing
    c = Currency(code, 8, 0, code, CurrencyType.CRYPTO)
    Currency.register(c, overwrite=True)
    return c


def _decimals_of(series: pd.Series, cap: int) -> int:
    vals = series.dropna().to_numpy()[:4000]
    max_d = 0
    for v in vals:
        s = f"{v:.{cap}f}".rstrip("0")
        if "." in s:
            max_d = max(max_d, len(s.split(".")[1]))
    return max_d


def build_instrument(base: str, symbol: str, df: pd.DataFrame) -> CryptoPerpetual:
    price_prec = max(_decimals_of(df["open"], 8), _decimals_of(df["close"], 8), 1)
    price_prec = min(price_prec, 9)
    size_prec = _decimals_of(df["volume"], 8)
    size_prec = min(max(size_prec, 1), 8)

    price_inc = 10.0**-price_prec
    size_inc = 10.0**-size_prec

    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(symbol), VENUE),
        raw_symbol=Symbol(symbol.replace("-PERP", "")),
        base_currency=get_or_create_currency(base),
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=price_prec,
        price_increment=Price(price_inc, precision=price_prec),
        size_precision=size_prec,
        size_increment=Quantity(size_inc, precision=size_prec),
        max_quantity=None,
        min_quantity=Quantity(size_inc, precision=size_prec),
        max_notional=None,
        min_notional=None,
        max_price=None,
        min_price=None,
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal("0.000200"),
        taker_fee=Decimal("0.000200"),
        ts_event=0,
        ts_init=0,
    )


def build_bars(symbol: str, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[Bar]:
    bar_type = BarType.from_str(f"{symbol}.{VENUE}-{TIMEFRAME_STR}-LAST-EXTERNAL")
    win = df.loc[start:end]
    price_prec = max(_decimals_of(df["open"], 8), _decimals_of(df["close"], 8), 1)
    price_prec = min(price_prec, 9)
    size_prec = min(max(_decimals_of(df["volume"], 8), 1), 8)

    bars: list[Bar] = []
    QUANTITY_MAX = 18_400_000_000.0
    n_clamped = 0
    o = win["open"].to_numpy(float)
    h = win["high"].to_numpy(float)
    l = win["low"].to_numpy(float)
    c = win["close"].to_numpy(float)
    v = win["volume"].to_numpy(float)
    for i, idx in enumerate(win.index):
        t = ts_ns(idx)
        vol = v[i]
        if vol > QUANTITY_MAX:
            vol = QUANTITY_MAX
            n_clamped += 1
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price(round(o[i], price_prec), precision=price_prec),
                high=Price(round(h[i], price_prec), precision=price_prec),
                low=Price(round(l[i], price_prec), precision=price_prec),
                close=Price(round(c[i], price_prec), precision=price_prec),
                volume=Quantity(round(float(vol), size_prec), precision=size_prec),
                ts_event=t,
                ts_init=t,
            )
        )
    if n_clamped:
        print(f"  {symbol}: {n_clamped} bars volume 超上限被截断 (无影响)")
    return bars


# ---------------------------------------------------------------------------
# 决策点构建（与 07 引擎完全同口径: dates[::REBAL], len(snap)<10 跳过）
# ---------------------------------------------------------------------------
def build_decisions(pdf: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                    rebal: int) -> dict[int, tuple[tuple[str, ...], dict[str, float]]]:
    w = pdf[(pdf["date"] >= start) & (pdf["date"] <= end)]
    dates = sorted(w["date"].unique())
    dds = dates[::rebal]
    decisions: dict[int, tuple[tuple[str, ...], dict[str, float]]] = {}
    for dd in dds:
        snap = w[w["date"] == dd].sort_values("pred", ascending=False)
        if len(snap) < 10:
            continue
        ranked = tuple(snap["base"].tolist())
        conf = {b: float(c) for b, c in zip(snap["base"], snap["conf_dn"])}
        decisions[ts_ns(pd.Timestamp(dd))] = (ranked, conf)
    return decisions


# ---------------------------------------------------------------------------
# 策略
# ---------------------------------------------------------------------------
class XSTopBottomNT(NTStrategy):
    """ML v2 横截面 Top3/Bottom3 — wf_preds_v2 离线信号 + NT 执行"""

    def __init__(
        self,
        decisions: dict[int, tuple[tuple[str, ...], dict[str, float]]],
        base2sym: dict[str, str],                     # preds base -> NT symbol
        instruments: dict[str, CryptoPerpetual],      # NT symbol -> instrument
        alive_count: dict[int, int] | None = None,    # ts_ns -> 该时刻有 bar 的币种数
        config=None,
        max_long: int = 3,
        max_short: int = 3,
        long_buffer: int = 5,
        short_buffer: int = 5,
        short_conf: float = 0.50,
        stoploss_price: float = 0.15,
        stake_pct: float = 0.10,
        leverage: float = 3.0,
        min_hold_bars: int = 32,
    ):
        super().__init__(config)
        self._decisions = decisions
        self._dd_list = sorted(decisions.keys())
        self._dd_ptr = 0
        self._base2sym = base2sym
        self._sym2base = {v: k for k, v in base2sym.items()}
        self._instruments = instruments
        self._max_long = max_long
        self._max_short = max_short
        self._long_buffer = long_buffer
        self._short_buffer = short_buffer
        self._short_conf = short_conf
        self._stoploss = stoploss_price
        self._stake_pct = stake_pct
        self._leverage = leverage
        self._min_hold_bars = min_hold_bars

        self._n_sub = len(instruments)
        self._alive = alive_count or {}
        self._cur_ts: int | None = None
        self._bucket: set[str] = set()
        self._px_snap: dict[str, float] = {}   # 当前时间戳桶内各币 close（决策定价用）
        self._decided: set[int] = set()
        self._last_dd = self._dd_list[-1] if self._dd_list else None

        # 同步影子账本: NT 的成交/仓位事件经消息队列异步送达（成交价在订单提交时已按
        # 当时订单簿确定, 但 on_position_* / cache / account 状态滞后送达）。横截面决策
        # 需要"同一决策内先平后开"的同步状态（与参考引擎 poss 字典等价）→ 提交订单时
        # 立即记账, 事件回调仅用于交易记录与对账。
        self._live: dict[str, PositionSide] = {}       # symbol -> 持仓方向（提交即记账）
        self._meta: dict[str, dict] = {}               # symbol -> {'side','entry_dd'}
        self._closing: set[str] = set()                # 平仓单在途的 symbol
        self._closing_meta: dict[str, dict] = {}       # symbol -> 在途平仓对应的 {'side','entry_dd'}
        self._pending: set[str] = set()                # 入场单在途的 symbol
        self._stops: dict[str, object] = {}
        self._exit_reason: dict[str, str] = {}
        self._closed_trades: list[dict] = []
        self._fill_log: list[dict] = []
        self._winddown = False

        self._n_entries = 0
        self._n_rejected_entry = 0
        self._n_rejected_stop = 0
        self._n_rank_exit = 0
        self._debug = bool(os.environ.get("XS_DEBUG"))
        self._dbg: list[str] = []
        self._final_open = 0
        self._final_unrealized = 0.0

    # --- 生命周期 ---
    def on_start(self):
        for symbol in self._instruments:
            bt = BarType.from_str(f"{symbol}.{VENUE}-{TIMEFRAME_STR}-LAST-EXTERNAL")
            self.subscribe_bars(bt)

    # --- 账户/仓位 ---
    def _equity(self) -> float:
        """净值 = 已实现余额 + 未实现盈亏（与 07 引擎 equity_at 口径一致）"""
        account = self.portfolio.account(VENUE)
        bal = account.balance_total(USDT).as_double()
        try:
            un = self.portfolio.unrealized_pnls(VENUE)
            if un:
                bal += sum(m.as_double() for m in un.values())
        except Exception:
            pass
        return bal

    def _size_for(self, symbol: str, notional: float) -> Quantity | None:
        inst = self._instruments[symbol]
        px = self._px_snap.get(symbol)
        if px is None or px <= 0:
            return None
        qty = inst.make_qty(notional / px)
        if qty.as_double() <= 0:
            return None
        if inst.min_quantity is not None and qty < inst.min_quantity:
            return None
        return qty

    def _min_hold_ok(self, symbol: str, dd: int) -> bool:
        m = self._meta.get(symbol)
        if m is None:
            return True
        return (dd - m["entry_dd"]) >= self._min_hold_bars * NS_BAR

    # --- bar 流: 按时间戳分桶, 桶满/桶边界触发横截面决策 ---
    def on_bar(self, bar):
        sym = bar.bar_type.instrument_id.symbol.value
        t = bar.ts_event
        px = bar.close.as_double()

        # 期末清扫: 最后决策点后仍开仓的币种(该时点缺 bar), 用其后首根 bar 收盘平掉
        if self._winddown and sym in self._live:
            self._exit_position(sym, "winddown")

        if t != self._cur_ts:
            # 桶边界: 上一时间戳的所有 bar 已处理完 → 触发该时刻的决策
            # (此时定价快照/订单簿仍是上一桶收盘, 全部币种统一按决策 bar close 成交)
            if self._cur_ts is not None:
                self._maybe_decide(self._cur_ts, self._bucket, self._px_snap)
                self._fire_stale(t)
            self._cur_ts = t
            self._bucket = set()
            self._px_snap = {}
        self._bucket.add(sym)
        self._px_snap[sym] = px
        # 桶满(该时刻应有 bar 的币种全部到齐) → 提前触发, 成交价同样为各币决策 bar close
        if len(self._bucket) >= self._alive.get(t, self._n_sub):
            self._maybe_decide(t, self._bucket, self._px_snap)

    def _maybe_decide(self, ts: int, present: set[str], px_snap: dict[str, float]):
        if ts in self._decisions and ts not in self._decided:
            self._decide(ts, present, px_snap)

    def _fire_stale(self, t: int):
        """时间已推进到 t, 但 (cur_ts, t) 之间还有未触发的决策点(该时刻无任何 bar)→空转标记"""
        while self._dd_ptr < len(self._dd_list) and self._dd_list[self._dd_ptr] < t:
            dd = self._dd_list[self._dd_ptr]
            self._dd_ptr += 1
            if dd not in self._decided:
                self._decided.add(dd)
                if dd == self._last_dd:
                    self._winddown = True

    # --- 横截面决策 ---
    def _decide(self, dd: int, present: set[str], px_snap: dict[str, float]):
        self._decided.add(dd)
        while self._dd_ptr < len(self._dd_list) and self._dd_list[self._dd_ptr] <= dd:
            self._dd_ptr += 1

        ranked_bases, conf = self._decisions[dd]
        ranked = [self._base2sym[b] for b in ranked_bases if b in self._base2sym]
        if not ranked:
            if dd == self._last_dd:
                self._winddown = True
            return
        n = len(ranked)
        top_buf = set(ranked[: self._long_buffer])
        bot_buf = set(ranked[n - self._short_buffer:])

        # === 1) 信号离场: 排名跌出缓冲区 / 空头置信度跌破门槛 (最小持有内不强平) ===
        for sym, side in list(self._live.items()):
            if sym not in present:
                continue  # 该决策点无 bar → 参考引擎同样跳过, 持有到下一决策点
            if not self._min_hold_ok(sym, dd):
                continue
            if side == PositionSide.LONG:
                if sym not in top_buf:
                    self._exit_position(sym, "rank_exit")
            else:
                if sym not in bot_buf:
                    self._exit_position(sym, "rank_exit")
                else:
                    c = conf.get(self._sym2base.get(sym, ""))
                    if c is not None and c < self._short_conf:
                        self._exit_position(sym, "conf_exit")

        # === 2) 期末强平（最后一个决策点, 不再开仓）===
        if dd == self._last_dd:
            self._winddown = True
            for sym in list(self._live.keys()):
                if sym in present:
                    self._exit_position(sym, "winddown")
            return

        # === 3) 开仓: 先多头后空头（与 07 引擎顺序一致）; _live 已同步扣除本决策平仓,
        #        因此与参考引擎一样支持"同决策先平后开" ===
        equity = self._equity()
        notional = equity * self._stake_pct * self._leverage
        n_long = sum(1 for s in self._live if self._live[s] == PositionSide.LONG)
        n_short = len(self._live) - n_long

        for sym in ranked[: self._max_long]:
            if n_long >= self._max_long:
                break
            if sym in self._live or sym in self._pending or sym not in present:
                continue
            qty = self._size_for(sym, notional)
            if qty is None:
                continue
            self._submit_entry(sym, OrderSide.BUY, qty, dd)
            n_long += 1

        for sym in ranked[n - self._max_short:]:
            if n_short >= self._max_short:
                break
            if sym in self._live or sym in self._pending or sym not in present:
                continue
            c = conf.get(self._sym2base.get(sym, ""))
            if c is None or c <= self._short_conf:
                continue  # 空头门槛: 仅 conf_dn > 0.50
            qty = self._size_for(sym, notional)
            if qty is None:
                continue
            self._submit_entry(sym, OrderSide.SELL, qty, dd)
            n_short += 1

        if self._debug:
            hold = {s: v.name[0] for s, v in self._live.items()}
            self._dbg.append(
                f"{pd.Timestamp(dd, unit='ns', tz='UTC')}  top5={ranked[:5]} bot5={ranked[-5:]} "
                f"hold={hold} entries+={self._n_entries}"
            )

    # --- 下单 ---
    def _submit_entry(self, symbol: str, side: OrderSide, qty: Quantity, dd: int):
        order = self.order_factory.market(
            instrument_id=self._instruments[symbol].id,
            order_side=side,
            quantity=qty,
            tags=["XS-ENTRY"],
        )
        self._pending.add(symbol)
        self._live[symbol] = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT
        self._meta[symbol] = {"side": self._live[symbol], "entry_dd": dd}
        self._n_entries += 1
        self.submit_order(order)

    def _exit_position(self, symbol: str, reason: str):
        """撤托管止损 + reduce-only 市价平仓（成交价=当根 bar 收盘）。同步从 _live
        记账移除, 使同一决策内释放的仓位立即可复用/反手（与参考引擎一致）"""
        if symbol in self._closing:
            return  # 平仓单已在途, 防止重复提交
        open_pos = self.cache.positions_open(instrument_id=self._instruments[symbol].id)
        if not open_pos:
            # 仓位实际已不存在（如止损刚触发而事件尚未送达）→ 仅同步记账
            self._live.pop(symbol, None)
            return
        self._closing.add(symbol)
        m = self._meta.get(symbol) or {}
        # 捕获"被平仓位"的元数据: 同决策反手时 _meta 会被新仓位覆盖, 平仓事件
        # 异步送达时需用它正确记录旧交易的 side/entry_dd
        self._closing_meta[symbol] = {"side": open_pos[0].side, "entry_dd": m.get("entry_dd")}
        self._live.pop(symbol, None)
        stop = self._stops.pop(symbol, None)
        if stop is not None and stop.is_open:
            self.cancel_order(stop)
        self._exit_reason[symbol] = reason
        self.close_position(open_pos[0], tags=["XS-EXIT"])

    # --- 事件 ---
    def on_order_filled(self, event):
        symbol = event.instrument_id.symbol.value
        if event.order_type == OrderType.STOP_MARKET:
            self._exit_reason[symbol] = "stoploss"
        if symbol in self._pending and event.order_side in (OrderSide.BUY, OrderSide.SELL):
            self._pending.discard(symbol)
            side = PositionSide.LONG if event.order_side == OrderSide.BUY else PositionSide.SHORT
            fill_px = event.last_px.as_double()
            if self._stoploss > 0:
                inst = self._instruments[symbol]
                trigger = fill_px * (1 - self._stoploss) if side == PositionSide.LONG \
                    else fill_px * (1 + self._stoploss)
                stop = self.order_factory.stop_market(
                    instrument_id=event.instrument_id,
                    order_side=OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY,
                    quantity=event.last_qty,
                    trigger_price=inst.make_price(trigger),
                    reduce_only=True,
                    tags=["XS-STOP"],
                )
                self._stops[symbol] = stop
                self.submit_order(stop)
            if len(self._fill_log) < 12:
                self._fill_log.append({
                    "symbol": symbol, "side": side.name, "fill_px": fill_px,
                    "qty": event.last_qty.as_double(), "ts": event.ts_event,
                })

    def on_position_opened(self, event):
        symbol = event.instrument_id.symbol.value
        meta = self._meta.get(symbol)
        if meta is not None:
            meta["side"] = event.side  # 以仓位事件为准校正方向

    def on_position_closed(self, event):
        symbol = event.instrument_id.symbol.value
        self._closing.discard(symbol)
        meta = self._closing_meta.pop(symbol, None)
        reason = self._exit_reason.pop(symbol, "unknown")
        if meta is not None:
            # 信号/期末平仓路径: side/entry_dd 取自平仓提交时的快照
            side, entry_dd = meta["side"], meta["entry_dd"]
            if symbol not in self._live:
                self._meta.pop(symbol, None)  # 无同决策再入场 → 清理记账
        else:
            # 止损触发路径: 平仓不经过 _exit_position, 从 _meta 取并同步清理
            m = self._meta.pop(symbol, None)
            side = m["side"] if m else None
            entry_dd = m["entry_dd"] if m else None
            self._live.pop(symbol, None)
            self._stops.pop(symbol, None)
        if reason == "rank_exit":
            self._n_rank_exit += 1
        pnl = event.realized_pnl.as_double() if event.realized_pnl is not None else 0.0
        self._closed_trades.append({
            "symbol": symbol,
            "side": side.name if side is not None else "?",
            "exit_reason": reason,
            "pnl_usdt": pnl,
            "return": event.realized_return,
            "closed_ts": event.ts_closed if event.ts_closed else event.ts_event,
            "opened_ts": event.ts_opened if event.ts_opened else None,
            "duration_min": (event.duration_ns / 60e9) if event.duration_ns else None,
        })

    def on_order_rejected(self, event):
        symbol = event.instrument_id.symbol.value
        if symbol in self._pending:
            self._pending.discard(symbol)
            self._live.pop(symbol, None)   # 入场被拒 → 撤销同步记账
            self._meta.pop(symbol, None)
            self._n_rejected_entry += 1
        elif symbol in self._closing:
            self._closing.discard(symbol)
            open_pos = self.cache.positions_open(instrument_id=self._instruments[symbol].id)
            if open_pos:
                # 平仓单被拒且仓位仍在（如保证金不足）→ 恢复记账, 下个决策点重试
                self._live[symbol] = open_pos[0].side
                self._exit_reason.pop(symbol, None)
                self._closing_meta.pop(symbol, None)
            # 仓位已平（被止损抢先, reduce-only 拒单）→ _closing_meta 留给随后的
            # PositionClosed 事件使用
        else:
            # 被拒的 reduce-only stop: 止损与信号平仓在同一根 bar 竞争, 仓位已平时正常发生
            self._n_rejected_stop += 1
        self._stops.pop(symbol, None)

    def on_stop(self):
        self._final_open = len(self._live)
        try:
            unrealized = self.portfolio.unrealized_pnls(VENUE)
            self._final_unrealized = sum(m.as_double() for m in unrealized.values()) if unrealized else 0.0
        except Exception:
            self._final_unrealized = 0.0
        if self._live:
            self.log.warning(f"结束时仍有 {len(self._live)} 个未平仓: {list(self._live)}")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def report(trades: list[dict], initial: float, start: pd.Timestamp, end: pd.Timestamp,
           n_instruments: int, n_bars: int, n_decisions: int, elapsed: float,
           fill_log: list[dict], n_entries: int, n_rejected: int, n_rejected_stop: int,
           final_open: int, final_unrealized: float) -> dict | None:
    if not trades:
        print("\n无交易 — 请检查信号/数据")
        return None

    tdf = pd.DataFrame(trades).sort_values("closed_ts")
    final = initial + tdf["pnl_usdt"].sum()

    eq = initial + tdf["pnl_usdt"].cumsum()
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min() * 100

    wins = (tdf["pnl_usdt"] > 0).sum()
    longs = tdf[tdf["side"] == "LONG"]
    shorts = tdf[tdf["side"] == "SHORT"]
    reasons = tdf["exit_reason"].value_counts().to_dict()

    days = (end - start).days

    print(f"\n{'=' * 64}")
    print("NautilusTrader ML v2 横截面 Top3/Bottom3 回测结果")
    print(f"{'=' * 64}")
    print(f"  时间范围        : {start.date()} -> {end.date()}  ({days} 天, {n_instruments} 币, {n_bars} bars, {n_decisions} 个决策点)")
    print(f"  初始资金        : {initial:.2f} USDT")
    print(f"  期末资金(已实现): {final:.3f} USDT")
    print(f"  总收益          : {final / initial * 100 - 100:+.2f}%")
    print(f"  交易数          : {len(tdf)}  (多头 {len(longs)} / 空头 {len(shorts)})")
    print(f"  胜率            : {wins / len(tdf) * 100:.1f}%")
    print(f"  多头收益        : {longs['pnl_usdt'].sum():+.2f} USDT ({len(longs)} 笔)")
    print(f"  空头收益        : {shorts['pnl_usdt'].sum():+.2f} USDT ({len(shorts)} 笔)")
    print(f"  最大回撤        : {max_dd:.2f}%")
    print(f"  离场原因        : {reasons}")
    print(f"  平均持仓时长    : {tdf['duration_min'].dropna().mean():.0f} 分钟")
    print(f"  入场信号数      : {n_entries}  (被拒: {n_rejected})")
    if n_rejected_stop:
        print(f"  竞争被拒止损单  : {n_rejected_stop} 个 (止损与信号平仓同 bar 竞争, 仓位已平, 无影响)")
    if final_open:
        print(f"  期末未平仓      : {final_open} 个, 未实现盈亏 {final_unrealized:+.3f} USDT (未计入上表)")

    # --- 分月收益（月内已实现盈亏 / 月初净值）---
    ts = pd.to_datetime(tdf["closed_ts"], unit="ns", utc=True).dt.tz_convert(None)
    tdf["month"] = ts.dt.to_period("M")
    print(f"\n  {'月份':>8s} {'交易':>5s} {'多':>4s} {'空':>4s} {'盈亏 USDT':>12s} {'月收益%':>9s} {'月末净值':>10s}")
    print("  " + "-" * 60)
    eq_prev = initial
    for m, g in tdf.groupby("month"):
        pnl = g["pnl_usdt"].sum()
        nl = (g["side"] == "LONG").sum()
        ns = (g["side"] == "SHORT").sum()
        eq_end = eq_prev + pnl
        print(f"  {str(m):>8s} {len(g):>5d} {nl:>4d} {ns:>4d} {pnl:>+12.3f} {pnl / eq_prev * 100:>+9.2f}% {eq_end:>10.2f}")
        eq_prev = eq_end

    print("\n  首批成交样本（验证决策 bar 收盘成交 ≈ 参考引擎下一根开盘）:")
    for f in fill_log:
        print(f"    {pd.Timestamp(f['ts'], unit='ns', tz='UTC')}  {f['symbol']:18s} "
              f"{f['side']:5s} px={f['fill_px']:.6g} qty={f['qty']:.6g}")

    return {
        "final_balance": final,
        "return_pct": final / initial * 100 - 100,
        "trades": len(tdf),
        "longs": len(longs),
        "shorts": len(shorts),
        "win_rate_pct": wins / len(tdf) * 100,
        "max_dd_pct": max_dd,
    }


# ---------------------------------------------------------------------------
# 对照: 07 pandas 引擎同窗口（funding 关闭, 口径对齐）
# ---------------------------------------------------------------------------
def run_reference(start: pd.Timestamp, end: pd.Timestamp, args) -> None:
    spec = importlib.util.spec_from_file_location("ref07", REF_ENGINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REBAL = args.rebal
    mod.MAX_L = args.max_long
    mod.MAX_S = args.max_short
    mod.BUFFER = args.long_buffer
    mod.SL = args.stoploss_pct
    mod.MIN_HOLD = args.min_hold

    pdf = pd.read_feather(PRED_FILE)
    pdf = pdf[(pdf["date"] >= start) & (pdf["date"] <= end)]
    ohlcv = {}
    for prefix, _ in load_pairs(args.pairs):
        f = DATA_DIR / f"{prefix}-15m-futures.feather"
        if f.exists():
            df = pd.read_feather(f).set_index("date")[["open", "high", "low", "close"]].loc[start:end]
            if len(df):
                ohlcv[prefix] = df
    trs, cap = mod.backtest(pdf, ohlcv, {}, args.short_conf)  # funding={} → 不含资金费
    tdf = pd.DataFrame(trs) if trs else pd.DataFrame()
    print(f"\n{'=' * 64}")
    print("对照: 07_full_ls_backtest.py pandas 引擎（同窗口, 无资金费）")
    print(f"{'=' * 64}")
    print(f"  总收益: {(cap / 150 - 1) * 100:+.2f}%   期末资金: {cap:.2f} USDT")
    if len(tdf):
        nl = (tdf["s"] == "long").sum()
        print(f"  交易数: {len(tdf)}  (多头 {nl} / 空头 {len(tdf) - nl})")
        print(f"  离场原因: {tdf['e'].value_counts().to_dict()}")
    print("  规则差异提醒: 07 空头跌出 Bottom3 即离场(硬编码), 本脚本默认 Bottom5 缓冲;")
    print("                07 长仓缓冲 = Top5 ∪ Bottom5; 07 止损只在决策 bar 检查(每 8h)")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=None, help="只用前 N 个币（验证用）")
    ap.add_argument("--start", type=str, default="2023-01-01")
    ap.add_argument("--end", type=str, default="2023-06-01")
    ap.add_argument("--balance", type=float, default=150.0)
    ap.add_argument("--stake-pct", type=float, default=0.10, help="保证金占净值比例")
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--max-long", type=int, default=3)
    ap.add_argument("--max-short", type=int, default=3)
    ap.add_argument("--long-buffer", type=int, default=5, help="多头缓冲区 Top N")
    ap.add_argument("--short-buffer", type=int, default=5, help="空头缓冲区 Bottom N")
    ap.add_argument("--short-conf", type=float, default=0.50, help="空头 conf_dn 门槛")
    ap.add_argument("--rebal", type=int, default=32, help="决策间隔（15m bar 数）")
    ap.add_argument("--min-hold", type=int, default=32, help="最小持有（15m bar 数）")
    ap.add_argument("--stoploss-pct", type=float, default=0.15, help="价格口径止损")
    ap.add_argument("--log-level", type=str, default="ERROR")
    ap.add_argument("--compare", action="store_true", help="结束后跑 07 pandas 引擎对照")
    ap.add_argument("--save", type=str, default=str(ROOT / "user_data" / "ml_v2" / "nt_xs_trades.feather"))
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    # bars 多加载 2h: 保证最后一个决策点(可能恰在窗口末端)的桶边界能触发
    bar_end = end + pd.Timedelta(hours=2)

    t0 = time.time()
    pairs = load_pairs(args.pairs)
    print(f"加载预测 + {len(pairs)} 币 K线 ...")

    pdf = pd.read_feather(PRED_FILE)
    decisions = build_decisions(pdf, start, end, args.rebal)
    print(f"预测窗口: {len(pdf[(pdf['date'] >= start) & (pdf['date'] <= end)])} 行, "
          f"决策点 {len(decisions)} 个 (每 {args.rebal} 根 15m bar)")

    instruments: list[CryptoPerpetual] = []
    inst_map: dict[str, CryptoPerpetual] = {}
    base2sym: dict[str, str] = {}
    alive_count: dict[int, int] = {}   # ts_ns -> 该时刻有 bar 的币种数（桶满触发用）
    all_bars: list[Bar] = []
    skipped = []

    for prefix, symbol in pairs:
        f = DATA_DIR / f"{prefix}-15m-futures.feather"
        try:
            df = pd.read_feather(f).set_index("date")
        except Exception as e:
            skipped.append((prefix, str(e)))
            continue
        win = df.loc[start:bar_end]
        if win.empty:
            skipped.append((prefix, "窗口内无数据"))
            continue
        inst = build_instrument(prefix.split("_")[0], symbol, win)
        instruments.append(inst)
        inst_map[symbol] = inst
        base2sym[prefix] = symbol
        for idx in win.index:
            k = ts_ns(idx)
            alive_count[k] = alive_count.get(k, 0) + 1
        bars = build_bars(symbol, win, start, bar_end)
        all_bars.extend(bars)
        print(f"  {symbol:18s} {len(bars):6d} bars  "
              f"price_prec={inst.price_precision} size_prec={inst.size_precision}")

    if skipped:
        print(f"  跳过: {skipped}")

    all_bars.sort(key=lambda b: b.ts_init)
    # 同一时间戳内随机打乱 bar 顺序(固定种子): 桶边界触发时, 触发币的订单簿已含下一根
    # bar(其成交会晚一根), 随机化使该偏差在各币种间均匀分布而不是固定砸在第一个币上
    rng = random.Random(42)
    grouped: list[Bar] = []
    for _, grp in itertools.groupby(all_bars, key=lambda b: b.ts_init):
        g = list(grp)
        rng.shuffle(g)
        grouped.extend(g)
    all_bars = grouped
    print(f"共 {len(all_bars)} bars, {len(instruments)} instruments — 构建引擎...")

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="XSML-BACKTEST-001",
            logging=LoggingConfig(log_level=args.log_level),
        )
    )
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(args.balance, USDT)],
        default_leverage=decimal.Decimal(str(args.leverage)),
        bar_execution=True,
    )
    for inst in instruments:
        engine.add_instrument(inst)
    engine.add_data(all_bars)

    strategy = XSTopBottomNT(
        decisions=decisions,
        base2sym=base2sym,
        instruments=inst_map,
        alive_count=alive_count,
        max_long=args.max_long,
        max_short=args.max_short,
        long_buffer=args.long_buffer,
        short_buffer=args.short_buffer,
        short_conf=args.short_conf,
        stoploss_price=args.stoploss_pct,
        stake_pct=args.stake_pct,
        leverage=args.leverage,
        min_hold_bars=args.min_hold,
    )
    engine.add_strategy(strategy)

    print("运行回测...")
    t1 = time.time()
    engine.run()
    run_elapsed = time.time() - t1

    # 引擎侧订单状态诊断
    orders = engine.trader.generate_orders_report()
    if len(orders) and "status" in orders:
        status_counts = orders["status"].value_counts().to_dict()
        abnormal = {k: v for k, v in status_counts.items()
                    if k not in ("FILLED", "CANCELED", "ACCEPTED")}
        if abnormal:
            print(f"  ⚠️ 异常订单状态: {abnormal}")

    result = report(
        trades=strategy._closed_trades,
        initial=args.balance,
        start=start,
        end=end,
        n_instruments=len(instruments),
        n_bars=len(all_bars),
        n_decisions=len(decisions),
        elapsed=run_elapsed,
        fill_log=strategy._fill_log,
        n_entries=strategy._n_entries,
        n_rejected=strategy._n_rejected_entry,
        n_rejected_stop=strategy._n_rejected_stop,
        final_open=strategy._final_open,
        final_unrealized=strategy._final_unrealized,
    )

    if args.save and strategy._closed_trades:
        out = pd.DataFrame(strategy._closed_trades)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        out.to_feather(args.save)
        print(f"\n  交易明细已保存: {args.save}")

    if strategy._debug:
        print("\n=== 决策 trace ===")
        for line in strategy._dbg:
            print(line)

    engine.dispose()

    if args.compare:
        run_reference(start, end, args)

    print(f"\n总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

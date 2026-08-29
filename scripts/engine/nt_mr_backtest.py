# -*- coding: utf-8 -*-
"""NautilusTrader 1.231.0 — MR 均值回归策略回测（freqtrade MeanReversionStrategy 移植）

策略规则（与 user_data/strategies/MeanReversionStrategy.py 一致）:
  指标: BB(20, 2σ, typical price) + RSI(14, Wilder) + EMA200   [talib / technical.qtpylib 预计算]
  多头入场: close < BB_lower & RSI < 30 & close > EMA200 & volume > 0
  空头入场: close > BB_upper & RSI > 70 & close < EMA200 & volume > 0
  离场:     close 回到 BB 中轨（long: close >= mid, short: close <= mid）
  止损:     入场价反向 10%（stop-market, reduce-only）  [freqtrade stoploss -0.30 保证金 / 3x = 10% 价格]
  仓位:     净值 10% 保证金 × 3x 杠杆 = 净值 30% 名义仓位，最大 4 并发
  费用:     maker=taker=0.02%（与 config.json fee=0.0002 一致）

成交时机（实测验证）:
  NautilusTrader 市价单在 on_bar(bar N) 中提交后，立即按 bar N 的 close 成交。
  该数据 15m 连续无断档, close[N] ≈ open[N+1]（49% 的 bar 有微小差异, 但相对幅度
  中位数=0, p95 < 0.03%）, 因此与 freqtrade "信号 bar 收盘出信号 → 下一根 bar 开盘
  成交" 的语义基本等价。止损单由后续 bar 的 high/low 触发，与 freqtrade 用 candle
  low/high 检查 stoploss 一致。

已知差异（相对 freqtrade 同窗口回测）:
  - freqtrade 合约回测含资金费率（NT 回测不含）→ 2023H1 多头付资金费/空头收资金费,
    使 freqtrade 多头收益略低、空头亏损略小
  - freqtrade 结束时强制平仓未平仓位并计入收益; NT 单独列出未实现盈亏
  - 指标预热: 两者都用 2400 根 15m startup candles（EMA/RSI 种子差异在窗口起点已衰减到 ~1e-25）

freqtrade 同窗口基准（config.json, 20230101-20230601, 27 币, 实测）:
  +0.86% | 222 trades (113L/109S) | win 64.9% | max DD 5.99% | avg duration 4:11 | Sharpe(daily) 0.46
  （其中 WLD/TIA/SEI 三个币 2023-06 前无期货数据, 实际有效 24 币, 与本脚本一致）

用法:
  python scripts/engine/nt_mr_backtest.py                      # 27 币, 2023-01-01 ~ 2023-06-01
  python scripts/engine/nt_mr_backtest.py --pairs 5            # 前 5 币快速验证
  python scripts/engine/nt_mr_backtest.py --start 2023-01-01 --end 2023-06-01
"""
from __future__ import annotations

import argparse
import decimal
import json
import time
from decimal import Decimal
from pathlib import Path

import pandas as pd
import talib
from technical import qtpylib

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
CONFIG_PATH = ROOT / "config.json"

VENUE = Venue("BINANCE")
TIMEFRAME_STR = "15-MINUTE"

# freqtrade startup_candle_count=2400 根 15m ≈ 25 天预热（EMA200/RSI 收敛）
WARMUP_BARS = 2400
BAR_SEC = 15 * 60


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def ts_ns(dt: pd.Timestamp) -> int:
    """UTC 时间戳 → 纳秒（与 Bar.ts_event 同一公式，保证 lookup key 一致）"""
    return int(dt.timestamp() * 1e9)


def load_pairs(pairs_limit: int | None) -> list[tuple[str, str]]:
    """返回 [(pair_file_prefix, symbol), ...] 例如 ("ETH_USDT_USDT", "ETHUSDT-PERP")"""
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    whitelist = cfg["exchange"]["pair_whitelist"]
    out = []
    for p in whitelist:
        base = p.split("/")[0]
        prefix = f"{base}_USDT_USDT"
        out.append((prefix, f"{base}USDT-PERP"))
    if pairs_limit:
        out = out[:pairs_limit]
    return out


def get_or_create_currency(code: str) -> Currency:
    """优先用 nautilus_trader.model.currencies 预定义币种，否则注册自定义 crypto Currency"""
    import nautilus_trader.model.currencies as predefined

    existing = getattr(predefined, code, None)
    if existing is not None:
        return existing
    c = Currency(code, 8, 0, code, CurrencyType.CRYPTO)  # iso4217=0, CRYPTO
    Currency.register(c, overwrite=True)
    return c


def _decimals_of(series: pd.Series, cap: int) -> int:
    """推断数据里最大的小数位数（抽样前 4000 行）"""
    vals = series.dropna().to_numpy()[:4000]
    max_d = 0
    for v in vals:
        s = f"{v:.{cap}f}".rstrip("0")
        if "." in s:
            max_d = max(max_d, len(s.split(".")[1]))
    return max_d


# ---------------------------------------------------------------------------
# 指标预计算（与 freqtrade populate_indicators 完全相同路径）
# ---------------------------------------------------------------------------
def precompute_indicators(pair_prefix: str, ind_start: pd.Timestamp) -> pd.DataFrame:
    f = DATA_DIR / f"{pair_prefix}-15m-futures.feather"
    df = pd.read_feather(f).set_index("date")
    df = df.loc[ind_start:]
    c = df["close"].astype(float)

    # freqtrade: bollinger_bands(typical_price(df), window=20, stds=2)
    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df.astype(float)), window=20, stds=2)
    df["bb_lower"] = bb["lower"]
    df["bb_mid"] = bb["mid"]
    df["bb_upper"] = bb["upper"]

    df["rsi"] = talib.RSI(c.to_numpy(), timeperiod=14)
    df["ema200"] = talib.EMA(c.to_numpy(), timeperiod=200)
    return df


def build_lookup(df: pd.DataFrame, start: pd.Timestamp) -> dict[int, tuple]:
    """{ts_ns: (bb_lower, bb_mid, bb_upper, rsi, ema200)} — 只保留回测窗口内的行"""
    win = df.loc[start:]
    lookup: dict[int, tuple] = {}
    cols = ["bb_lower", "bb_mid", "bb_upper", "rsi", "ema200", "close", "volume"]
    for row in win[cols].itertuples():
        key = ts_ns(row.Index)
        lookup[key] = (row.bb_lower, row.bb_mid, row.bb_upper, row.rsi, row.ema200)
    return lookup


# ---------------------------------------------------------------------------
# Bar 与 Instrument 构建
# ---------------------------------------------------------------------------
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
        max_quantity=None,          # 不设上限，避免大仓位被拒
        min_quantity=Quantity(size_inc, precision=size_prec),
        max_notional=None,
        min_notional=None,
        max_price=None,
        min_price=None,
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal("0.000200"),   # 与 config.json fee=0.0002 对齐
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
    QUANTITY_MAX = 18_400_000_000.0  # NautilusTrader Quantity 上限 18_446_744_073, 留余量
    n_clamped = 0
    for idx, row in win.iterrows():
        t = ts_ns(idx)
        vol = float(row["volume"])
        if vol > QUANTITY_MAX:  # 策略只用 volume>0 判断, 截断无害
            vol = QUANTITY_MAX
            n_clamped += 1
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price(round(float(row["open"]), price_prec), precision=price_prec),
                high=Price(round(float(row["high"]), price_prec), precision=price_prec),
                low=Price(round(float(row["low"]), price_prec), precision=price_prec),
                close=Price(round(float(row["close"]), price_prec), precision=price_prec),
                volume=Quantity(round(float(vol), size_prec), precision=size_prec),
                ts_event=t,
                ts_init=t,
            )
        )
    if n_clamped:
        print(f"  {symbol}: {n_clamped} bars volume 超上限被截断 (策略仅用 volume>0, 无影响)")
    return bars


# ---------------------------------------------------------------------------
# 策略
# ---------------------------------------------------------------------------
class MeanReversionNT(NTStrategy):
    """布林+RSI 均值回归 — freqtrade MeanReversionStrategy 的 NT 移植"""

    def __init__(
        self,
        lookups: dict[str, dict[int, tuple]],        # {symbol: {ts_ns: (bb_l, bb_m, bb_u, rsi, ema)}}
        instruments: dict[str, CryptoPerpetual],      # {symbol: instrument}
        config=None,
        rsi_entry: int = 30,
        stoploss_price: float = 0.10,
        stake_pct: float = 0.10,
        leverage: float = 3.0,
        max_open: int = 4,
    ):
        super().__init__(config)
        self._lookups = lookups
        self._instruments = instruments
        self._rsi_entry = rsi_entry
        self._stoploss = stoploss_price
        self._stake_pct = stake_pct
        self._leverage = leverage
        self._max_open = max_open

        self._positions: dict[str, PositionSide] = {}   # symbol -> side
        self._stops: dict[str, object] = {}             # symbol -> working stop order
        self._pending: set[str] = set()                 # 已提交未成交的入场单
        self._exit_reason: dict[str, str] = {}          # symbol -> 'signal' | 'stoploss'
        self._closed_trades: list[dict] = []            # 平仓记录
        self._fill_log: list[dict] = []                 # 调试用成交日志
        self._n_signals = 0
        self._n_rejected_entry = 0
        self._final_open = 0
        self._final_unrealized = None

    # --- 生命周期 ---
    def on_start(self):
        for symbol in self._instruments:
            bt = BarType.from_str(f"{symbol}.{VENUE}-{TIMEFRAME_STR}-LAST-EXTERNAL")
            self.subscribe_bars(bt)

    # --- 工具 ---
    def _equity(self) -> float:
        account = self.portfolio.account(VENUE)
        bal = account.balance_total(USDT)
        return bal.as_double()

    def _size_for(self, symbol: str, price: float) -> Quantity | None:
        inst = self._instruments[symbol]
        notional = self._equity() * self._stake_pct * self._leverage
        raw = notional / price
        qty = inst.make_qty(raw)
        if qty.as_double() <= 0:
            return None
        if inst.min_quantity is not None and qty < inst.min_quantity:
            return None
        return qty

    # --- 事件 ---
    def on_bar(self, bar):
        symbol = bar.bar_type.instrument_id.symbol.value
        row = self._lookups.get(symbol, {}).get(bar.ts_event)
        if row is None:
            return
        bb_lower, bb_mid, bb_upper, rsi, ema200 = row
        if pd.isna(bb_lower) or pd.isna(rsi) or pd.isna(ema200):
            return

        close = bar.close.as_double()
        state = self._positions.get(symbol)

        # 有入场单在路上（等下一根 bar 成交）→ 本 bar 不做决策
        if symbol in self._pending:
            return

        # === 持仓管理: 中轨离场（止损由 stop-market 托管，触发时走 on_position_closed） ===
        if state is not None:
            exit_long = state == PositionSide.LONG and close >= bb_mid
            exit_short = state == PositionSide.SHORT and close <= bb_mid
            if exit_long or exit_short:
                self._exit_position(symbol)
            return

        # === 入场 ===
        if len(self._positions) + len(self._pending) >= self._max_open:
            self._n_rejected_entry += 1
            return
        if bar.volume.as_double() <= 0:
            return

        long_sig = close < bb_lower and rsi < self._rsi_entry and close > ema200
        short_sig = close > bb_upper and rsi > 100 - self._rsi_entry and close < ema200
        if not (long_sig or short_sig):
            return

        qty = self._size_for(symbol, close)
        if qty is None:
            self._n_rejected_entry += 1
            return

        order = self.order_factory.market(
            instrument_id=self._instruments[symbol].id,
            order_side=OrderSide.BUY if long_sig else OrderSide.SELL,
            quantity=qty,
            tags=["MR-ENTRY"],
        )
        self._pending.add(symbol)
        self._n_signals += 1
        self.submit_order(order)

    def _exit_position(self, symbol: str):
        """撤掉托管止损 + 市价平仓（当根 bar 收盘价成交）"""
        stop = self._stops.pop(symbol, None)
        if stop is not None and stop.is_open:  # property, not method
            self.cancel_order(stop)
        open_pos = self.cache.positions_open(instrument_id=self._instruments[symbol].id)
        if open_pos:
            self._exit_reason[symbol] = "signal"
            self.close_position(open_pos[0], tags=["MR-EXIT"])

    def on_order_filled(self, event):
        symbol = event.instrument_id.symbol.value
        if event.order_type == OrderType.STOP_MARKET:
            self._exit_reason[symbol] = "stoploss"
        if symbol in self._pending and event.order_side in (OrderSide.BUY, OrderSide.SELL):
            # 入场成交（下一根 bar 开盘价）
            self._pending.discard(symbol)
            if symbol in self._positions:
                return
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
                    tags=["MR-STOP"],
                )
                self._stops[symbol] = stop
                self.submit_order(stop)
            if len(self._fill_log) < 10:
                self._fill_log.append({
                    "symbol": symbol, "side": side.name, "fill_px": fill_px,
                    "qty": event.last_qty.as_double(), "ts": event.ts_event,
                })

    def on_position_opened(self, event):
        self._positions[event.instrument_id.symbol.value] = event.side

    def on_position_closed(self, event):
        symbol = event.instrument_id.symbol.value
        side = self._positions.pop(symbol, None)
        self._stops.pop(symbol, None)
        reason = self._exit_reason.pop(symbol, "unknown")
        pnl = event.realized_pnl.as_double() if event.realized_pnl is not None else 0.0
        self._closed_trades.append({
            "symbol": symbol,
            "side": side.name if side is not None else "?",  # PositionClosed 事件里 side 已变 FLAT
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
            self._n_rejected_entry += 1
        self._stops.pop(symbol, None)  # 被拒的 reduce-only stop（仓位已被信号单平掉时正常发生）

    def on_stop(self):
        # 引擎停止时快照期末状态（不再有事件）
        self._final_open = len(self._positions)
        try:
            unrealized = self.portfolio.unrealized_pnls(VENUE)
            self._final_unrealized = sum(m.as_double() for m in unrealized.values()) if unrealized else 0.0
        except Exception:
            self._final_unrealized = 0.0
        if self._pending:
            self.log.warning(f"结束时仍有 {len(self._pending)} 个入场单未成交")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def report(trades: list[dict], initial: float, start: pd.Timestamp, end: pd.Timestamp,
           n_instruments: int, n_bars: int, elapsed: float, fill_log: list[dict],
           n_signals: int, n_rejected: int, final_open: int = 0,
           final_unrealized: float = 0.0) -> dict | None:
    if not trades:
        print("\n无交易 — 请检查信号条件")
        return None

    tdf = pd.DataFrame(trades).sort_values("closed_ts")
    final = initial + tdf["pnl_usdt"].sum()

    # 净值曲线（已实现口径，与 freqtrade wallet balance 一致）→ 最大回撤
    eq = initial + tdf["pnl_usdt"].cumsum()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100
    dd_end_ts = tdf["closed_ts"].iloc[dd.idxmin()] if len(dd) else None

    wins = (tdf["pnl_usdt"] > 0).sum()
    longs = tdf[tdf["side"] == "LONG"]
    shorts = tdf[tdf["side"] == "SHORT"]
    n_stop = (tdf["exit_reason"] == "stoploss").sum() if "exit_reason" in tdf else 0

    # 日收益 Sharpe（已实现净值）
    eq_ser = pd.Series(eq.to_numpy(), index=pd.to_datetime(tdf["closed_ts"], unit="ns", utc=True))
    daily = initial + (eq_ser.resample("1D").last().ffill() - initial) + 0  # 每日已实现净值
    daily_ret = daily.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (365**0.5)) if len(daily_ret) > 2 and daily_ret.std() > 0 else float("nan")

    days = (end - start).days

    print(f"\n{'=' * 64}")
    print("NautilusTrader MR 回测结果")
    print(f"{'=' * 64}")
    print(f"  时间范围        : {start.date()} -> {end.date()}  ({days} 天, {n_instruments} 币, {n_bars} bars)")
    print(f"  初始资金        : {initial:.2f} USDT")
    print(f"  期末资金(已实现): {final:.3f} USDT")
    print(f"  总收益          : {final / initial * 100 - 100:+.2f}%")
    print(f"  交易数          : {len(tdf)}  (long {len(longs)} / short {len(shorts)})")
    print(f"  胜率            : {wins / len(tdf) * 100:.1f}%")
    print(f"  最大回撤        : {max_dd:.2f}%")
    if dd_end_ts is not None:
        print(f"    (回撤谷底     : {pd.Timestamp(dd_end_ts, unit='ns', tz='UTC')})")
    print(f"  日净值 Sharpe   : {sharpe:.2f}")
    print(f"  多头收益        : {longs['pnl_usdt'].sum():+.2f} USDT ({len(longs)} 笔)")
    print(f"  空头收益        : {shorts['pnl_usdt'].sum():+.2f} USDT ({len(shorts)} 笔)")
    print(f"  止损离场        : {n_stop} 笔, 信号离场: {len(tdf) - n_stop} 笔")
    print(f"  平均持仓时长    : {tdf['duration_min'].dropna().mean():.0f} 分钟")
    print(f"  入场信号数      : {n_signals}  (因并发上限/最小手数被拒: {n_rejected})")
    if final_open:
        print(f"  期末未平仓      : {final_open} 个, 未实现盈亏 {final_unrealized:+.3f} USDT (未计入上表)")
    print(f"  运行耗时        : {elapsed:.0f}s")

    print("\n  首笔成交样本（验证下一根 bar 开盘成交）:")
    for f in fill_log:
        print(f"    {pd.Timestamp(f['ts'], unit='ns', tz='UTC')}  {f['symbol']:16s} "
              f"{f['side']:5s} px={f['fill_px']:.6g} qty={f['qty']:.6g}")

    return {
        "final_balance": final,
        "return_pct": final / initial * 100 - 100,
        "trades": len(tdf),
        "win_rate_pct": wins / len(tdf) * 100,
        "max_dd_pct": max_dd,
        "sharpe_daily": sharpe,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=None, help="只用前 N 个币（验证用）")
    ap.add_argument("--start", type=str, default="2023-01-01")
    ap.add_argument("--end", type=str, default="2023-06-01")
    ap.add_argument("--balance", type=float, default=150.0, help="初始资金 USDT")
    ap.add_argument("--stake-pct", type=float, default=0.10, help="保证金占净值比例")
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--max-open", type=int, default=4)
    ap.add_argument("--stoploss-pct", type=float, default=0.10, help="价格口径止损")
    ap.add_argument("--rsi-entry", type=int, default=30)
    ap.add_argument("--log-level", type=str, default="ERROR")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    # 指标预热窗口（talib EMA/RSI 种子收敛，与 freqtrade startup_candle_count=2400 对齐）
    ind_start = start - pd.Timedelta(seconds=WARMUP_BARS * BAR_SEC + 2 * 86400)

    t0 = time.time()
    pairs = load_pairs(args.pairs)
    print(f"加载 {len(pairs)} 币数据 + 预计算指标 (预热自 {ind_start.date()}) ...")

    instruments: list[CryptoPerpetual] = []
    inst_map: dict[str, CryptoPerpetual] = {}
    lookups: dict[str, dict[int, tuple]] = {}
    all_bars: list[Bar] = []
    skipped = []

    for prefix, symbol in pairs:
        try:
            df = precompute_indicators(prefix, ind_start)
        except Exception as e:
            skipped.append((prefix, str(e)))
            continue
        if df.loc[start:end].empty:
            skipped.append((prefix, "窗口内无数据"))
            continue
        inst = build_instrument(prefix.split("_")[0], symbol, df)
        instruments.append(inst)
        inst_map[symbol] = inst
        lookups[symbol] = build_lookup(df, start)
        bars = build_bars(symbol, df, start, end)
        all_bars.extend(bars)
        print(f"  {symbol:16s} {len(bars):6d} bars  "
              f"price_prec={inst.price_precision} size_prec={inst.size_precision}")

    if skipped:
        print(f"  跳过: {skipped}")

    all_bars.sort(key=lambda b: b.ts_init)
    print(f"共 {len(all_bars)} bars, {len(instruments)} instruments — 构建引擎...")

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="MR-BACKTEST-001",
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

    strategy = MeanReversionNT(
        lookups=lookups,
        instruments=inst_map,
        rsi_entry=args.rsi_entry,
        stoploss_price=args.stoploss_pct,
        stake_pct=args.stake_pct,
        leverage=args.leverage,
        max_open=args.max_open,
    )
    engine.add_strategy(strategy)

    print("运行回测...")
    t1 = time.time()
    engine.run()
    run_elapsed = time.time() - t1

    # 引擎侧诊断: 被拒订单
    orders = engine.trader.generate_orders_report()
    if len(orders):
        status_counts = orders["status"].value_counts().to_dict() if "status" in orders else {}
        rejected = {k: v for k, v in status_counts.items() if k not in ("FILLED",)}
        if rejected:
            print(f"  订单状态: {status_counts}")

    result = report(
        trades=strategy._closed_trades,
        initial=args.balance,
        start=start,
        end=end,
        n_instruments=len(instruments),
        n_bars=len(all_bars),
        elapsed=run_elapsed,
        fill_log=strategy._fill_log,
        n_signals=strategy._n_signals,
        n_rejected=strategy._n_rejected_entry,
        final_open=strategy._final_open,
        final_unrealized=strategy._final_unrealized or 0.0,
    )

    # 引擎侧订单状态诊断
    orders = engine.trader.generate_orders_report()
    if len(orders) and "status" in orders:
        status_counts = orders["status"].value_counts().to_dict()
        abnormal = {k: v for k, v in status_counts.items()
                    if k not in ("FILLED", "CANCELED", "ACCEPTED")}
        if abnormal:
            print(f"  ⚠️ 异常订单状态: {abnormal}")
        n_accepted = status_counts.get("ACCEPTED", 0)
        if n_accepted:
            print(f"  (期末在途未成交订单: {n_accepted} 个 — 引擎停止时仍未到下一根 bar)")

    engine.dispose()

    # === 与 freqtrade 同窗口基准对比 ===
    print(f"\n{'=' * 64}")
    print(f"对比 freqtrade 同窗口回测 ({args.start} ~ {args.end}, config.json, 27 币)")
    print(f"{'=' * 64}")
    ft_window = {
        "return_pct": 0.86, "trades": 222, "win_rate_pct": 64.9, "max_dd_pct": 5.99,
    }
    if result:
        rows = [
            ("总收益 %", result["return_pct"], ft_window["return_pct"]),
            ("交易数", result["trades"], ft_window["trades"]),
            ("胜率 %", result["win_rate_pct"], ft_window["win_rate_pct"]),
            ("最大回撤 %", result["max_dd_pct"], ft_window["max_dd_pct"]),
        ]
        print(f"  {'指标':10s} {'NT':>10s} {'freqtrade':>10s}")
        for name, nt_v, ft_v in rows:
            print(f"  {name:10s} {nt_v:>10.2f} {ft_v:>10.2f}")
        trade_ratio = result["trades"] / ft_window["trades"]
        print(f"\n  交易数比 NT/FT = {trade_ratio:.2f}, 收益差 = {result['return_pct'] - ft_window['return_pct']:+.2f}pp")
        print("  残余差异来源: freqtrade 含资金费率(NT 不含) / 成交价 microstructure 差异 /")
        print("                freqtrade 期末强平未平仓(NT 单列未实现盈亏)")
    print("\n  复现 freqtrade 基准: python -m freqtrade backtesting -c config.json "
          f"-s MeanReversionStrategy --timerange {args.start.replace('-', '')}-{args.end.replace('-', '')} --cache none")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""ML 策略层回测: 信号处理 + 仓位管理 + 执行模拟 (轻量自研回测引擎, 不依赖 freqtrade)

输入
----
  user_data/ml/predictions.feather
      列: date, base, pred_ret, pred_direction, pred_confidence
      (date 对齐 15m K 线 bar 时间; pred_ret 为前瞻收益预测)
  user_data/data/binance/futures/{BASE}_USDT_USDT-15m-futures.feather
      列: date, open, high, low, close, volume

策略规则 (只做多)
----------------
1. 信号:   EMA(span=4) 平滑 pred_ret, 平滑值 > threshold 且 confidence > min_confidence
2. 仓位:   size = min(目标波动率/预测波动率, Kelly) * 相关性降权,
          再受 组合约束 (<=6 仓 / 单仓 <=20% / 总敞口 <=100%)
3. 入场:   信号 bar 收盘后挂限价单 close*(1-0.05%),
          下一根 bar low < 限价 -> 以 min(限价, 下根开盘) 成交, 否则作废
4. 离场:   自适应持有期, 初始 1h; 到期时信号仍同向则每次延长 1h, 最长 4h
5. 手续费: maker 0.02% 入场 + maker 0.02% 离场 = 0.04% RT

输出
----
  user_data/ml/backtest_trades.feather   交易明细
  user_data/ml/backtest_equity.feather   净值曲线 (逐 15m bar)
  user_data/ml/backtest_report.json      绩效报告 (收益/年化/Sharpe/回撤/胜率/盈亏比)

用法
----
  python scripts/ml/strategy_backtest.py                        # 真实 predictions 回测
  python scripts/ml/strategy_backtest.py --start 2025-01-01 --end 2025-06-01
  python scripts/ml/strategy_backtest.py --mock                 # 模拟信号冒烟测试(输出 *_mock 后缀)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 路径与常量
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/ml/ → 项目根
ML_DIR = ROOT / "user_data" / "ml"
DATA_DIR = ROOT / "user_data" / "data" / "binance" / "futures"
PRED_PATH = ML_DIR / "predictions.feather"

BAR_NS = 15 * 60 * 10 ** 9          # 一根 15m bar 的纳秒数
BAR_HOURS = 0.25

REQUIRED_PRED_COLS = ("date", "base", "pred_ret", "pred_confidence")


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    # ---- 信号 ----
    ema_span: int = 4               # 预测值 EMA 平滑窗口
    threshold: float = 0.001        # 平滑后 pred_ret 阈值 (0.1%)
    min_confidence: float = 0.6     # 最低置信度
    # ---- 仓位管理 ----
    target_vol: float = 0.01        # 单笔目标波动率 (初始持有期内)
    vol_window: int = 96            # 已实现波动率窗口 (96 根 15m = 24h)
    kelly_min_trades: int = 30      # Kelly 生效所需最少历史交易数
    kelly_lookback: int = 100       # Kelly 使用的近段交易数
    max_positions: int = 6          # 最大并发仓位数
    max_pos_frac: float = 0.20      # 单仓位占净值上限
    max_gross: float = 1.00         # 总敞口上限 (含挂单)
    min_pos_frac: float = 0.01      # 低于此仓位放弃开仓
    corr_window: int = 672          # 相关性窗口 (672 根 15m = 7 天)
    corr_threshold: float = 0.8     # 高相关阈值
    corr_dedup_min: int = 3         # 已持有 >=3 个高相关币时新信号降权
    corr_size_factor: float = 0.5   # 相关性去重降权系数
    corr_min_overlap: int = 200     # 相关性所需最少重叠样本
    # ---- 执行 ----
    entry_limit_offset: float = 0.0005   # 入场限价: 收盘价 - 0.05%
    fee_maker: float = 0.0002            # 单边 maker 费率
    hold_init_bars: int = 4              # 初始持有 4 根 = 1h
    hold_extend_bars: int = 4            # 每次延长 4 根 = 1h
    hold_max_bars: int = 16              # 最长持有 16 根 = 4h
    # ---- 回测 ----
    initial_equity: float = 1.0


# --------------------------------------------------------------------------
# 1. 信号处理
# --------------------------------------------------------------------------
def smooth_predictions(pred_series: pd.Series, ema_span: int = 4) -> pd.Series:
    """EMA 平滑预测值，减少噪声翻转"""
    return pred_series.ewm(span=ema_span).mean()


def filter_signals(pred: pd.Series, confidence: pd.Series,
                   threshold: float = 0.001,
                   min_confidence: float = 0.6) -> pd.Series:
    """只保留高置信度信号 (NaN 自动判 False)"""
    return (pred > threshold) & (confidence > min_confidence)


# --------------------------------------------------------------------------
# 2. 仓位管理
# --------------------------------------------------------------------------
def kelly_fraction(trade_rets) -> float:
    """Kelly 公式: f* = (胜率*赔率 - 败率)/赔率

    trade_rets: 该品种近段净收益率序列。
    样本不足 / 无法估计时返回 inf (不构成约束, 由波动率与硬上限接管);
    f* <= 0 时返回 0 (无正期望, 不开新仓)。
    """
    r = np.asarray(trade_rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return math.inf
    wins, losses = r[r > 0], r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return math.inf
    p = len(wins) / len(r)
    q = 1.0 - p
    avg_loss = abs(losses.mean())
    if avg_loss <= 0:
        return math.inf
    b = wins.mean() / avg_loss           # 赔率 = 平均盈利/平均亏损
    if b <= 0:
        return math.inf
    f = (p * b - q) / b
    return max(f, 0.0)


def volatility_size(bar_vol: float, cfg: BacktestConfig) -> float:
    """波动率调整仓位: size = 目标波动率 / 初始持有期内的预测波动率"""
    if not np.isfinite(bar_vol) or bar_vol <= 0:
        return math.inf  # 无法估计时交给硬上限
    horizon_vol = bar_vol * math.sqrt(cfg.hold_init_bars)
    return cfg.target_vol / horizon_vol


def high_corr_count(new_base: str, held_bases, sds: dict, ptrs: dict,
                    t_ns: int, cfg: BacktestConfig) -> int:
    """统计当前已持有币中与 new_base 滚动相关性 > corr_threshold 的个数"""
    if len(held_bases) < cfg.corr_dedup_min:
        return 0
    start = t_ns - cfg.corr_window * BAR_NS
    sd_new = sds[new_base]
    i0 = np.searchsorted(sd_new.ts, start)
    ts_a = sd_new.ts[i0:ptrs[new_base]]
    r_a = sd_new.ret[i0:ptrs[new_base]]
    cnt = 0
    for hb in held_bases:
        if hb == new_base:
            continue
        sd = sds[hb]
        j0 = np.searchsorted(sd.ts, start)
        ts_b = sd.ts[j0:ptrs[hb]]
        r_b = sd.ret[j0:ptrs[hb]]
        inter, ia, ib = np.intersect1d(ts_a, ts_b, assume_unique=True,
                                       return_indices=True)
        if len(inter) < cfg.corr_min_overlap:
            continue
        c = np.corrcoef(r_a[ia], r_b[ib])[0, 1]
        if np.isfinite(c) and c > cfg.corr_threshold:
            cnt += 1
    return cnt


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass
class SeriesData:
    """单个 base 的 K 线 + 对齐后的信号数组"""
    base: str
    ts: np.ndarray          # int64 纳秒, 升序
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    ret: np.ndarray         # 15m 收益率 (首行 NaN)
    vol: np.ndarray         # 滚动已实现波动率 (warmup NaN)
    pred: np.ndarray        # 平滑后 pred_ret, 对齐 K 线行 (缺失 NaN)
    conf: np.ndarray        # 置信度, 对齐 K 线行
    sig_mask: np.ndarray    # bool, filter_signals 结果

    @property
    def n(self) -> int:
        return len(self.ts)


@dataclass
class Position:
    base: str
    entry_ts: int               # 成交 bar 时间
    entry_row: int
    entry_price: float
    size_frac: float            # 占入场时净值的比例
    notional: float             # 入场时净值口径的名义市值
    signal_ts: int              # 产生信号的 bar 时间
    signal_strength: float      # 平滑后 pred_ret
    pred_raw: float
    conf: float
    exit_ts: int                # 当前计划离场/检查时间
    extensions: int = 0
    last_price: float = 0.0


def to_ns(s: pd.Series) -> np.ndarray:
    """UTC datetime Series -> int64 纳秒数组"""
    s = pd.to_datetime(s, utc=True)
    return s.to_numpy(dtype="datetime64[ns]").astype("int64")


# --------------------------------------------------------------------------
# 数据加载与信号对齐
# --------------------------------------------------------------------------
def kline_path(base: str) -> Path:
    return DATA_DIR / f"{base}_USDT_USDT-15m-futures.feather"


def discover_bases(pred_df: pd.DataFrame | None, only: set | None) -> list[str]:
    """真实模式取 predictions 里的 base; mock 模式取磁盘上有 15m K 线的 base"""
    if pred_df is not None:
        bases = sorted(pred_df["base"].unique().tolist())
    else:
        bases = sorted(p.stem.split("_USDT_USDT")[0]
                       for p in DATA_DIR.glob("*_USDT_USDT-15m-futures.feather"))
    if only:
        bases = [b for b in bases if b in only]
    return [b for b in bases if kline_path(b).exists()]


def load_klines(bases: list[str], start=None, end=None) -> dict[str, pd.DataFrame]:
    out = {}
    for b in bases:
        df = pd.read_feather(kline_path(b))
        df = df[["date", "open", "high", "low", "close"]].sort_values("date")
        if start is not None:
            # 回退留足波动率 warmup 窗口
            lo = pd.Timestamp(start, tz="UTC") - pd.Timedelta(nanoseconds=130 * BAR_NS)
            df = df[df["date"] >= lo]
        if end is not None:
            hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(nanoseconds=2 * 24 * 4 * BAR_NS)
            df = df[df["date"] <= hi]
        if len(df) > 0:
            out[b] = df.reset_index(drop=True)
    return out


def build_series_data(klines: dict[str, pd.DataFrame], pred_df: pd.DataFrame,
                      cfg: BacktestConfig) -> dict[str, SeriesData]:
    """加载 K 线, 把每个 base 的平滑预测/置信度/信号掩码对齐到 K 线行"""
    sds: dict[str, SeriesData] = {}
    for base, df in klines.items():
        c = df["close"].to_numpy(dtype=float)
        ts = to_ns(df["date"])
        ret = np.full(len(c), np.nan)
        ret[1:] = c[1:] / c[:-1] - 1.0
        vol = pd.Series(ret).rolling(cfg.vol_window, min_periods=cfg.vol_window).std().to_numpy()

        n = len(c)
        pred = np.full(n, np.nan)
        conf = np.full(n, np.nan)
        sub = pred_df[pred_df["base"] == base].sort_values("date")
        if len(sub) > 0:
            sm = smooth_predictions(sub["pred_ret"].reset_index(drop=True), cfg.ema_span)
            cf = sub["pred_confidence"].reset_index(drop=True)
            mask_sub = filter_signals(sm, cf, cfg.threshold, cfg.min_confidence)
            ts_sub = to_ns(sub["date"])
            pos = np.searchsorted(ts, ts_sub, side="left")
            ok = (pos < n) & (ts[np.clip(pos, 0, n - 1)] == ts_sub)
            rows = pos[ok]
            pred[rows] = sm.to_numpy()[ok]
            conf[rows] = cf.to_numpy()[ok]
        sig_mask = (pred > cfg.threshold) & np.isfinite(pred) & (conf > cfg.min_confidence)
        sig_mask = np.nan_to_num(sig_mask).astype(bool)

        sds[base] = SeriesData(base=base, ts=ts, open=df["open"].to_numpy(dtype=float),
                               high=df["high"].to_numpy(dtype=float),
                               low=df["low"].to_numpy(dtype=float),
                               close=c, ret=ret, vol=vol, pred=pred, conf=conf,
                               sig_mask=sig_mask)
    return sds


# --------------------------------------------------------------------------
# 4. 回测引擎 (按时间逐 bar)
# --------------------------------------------------------------------------
def run_backtest(sds: dict[str, SeriesData], cfg: BacktestConfig):
    """返回 (trades: list[dict], equity_records: list[dict], stats: dict)"""
    # 全局事件流: (bar时间, base, 行号), 按时间排序
    all_ts, all_base, all_row = [], [], []
    for bi, (b, sd) in enumerate(sds.items()):
        all_ts.append(sd.ts)
        all_base.append(np.full(sd.n, bi, dtype=np.int16))
        all_row.append(np.arange(sd.n, dtype=np.int32))
    base_list = list(sds.keys())
    ev_ts = np.concatenate(all_ts)
    ev_base = np.concatenate(all_base)
    ev_row = np.concatenate(all_row)
    order = np.lexsort((ev_base, ev_ts))
    ev_ts, ev_base, ev_row = ev_ts[order], ev_base[order], ev_row[order]
    timeline = np.unique(ev_ts)

    ptrs = {b: 0 for b in sds}          # 各 base 下一个未处理行
    pending: dict[str, dict] = {}       # base -> 挂单 (最多一张)
    open_pos: dict[str, Position] = {}
    trade_hist: dict[str, deque] = {b: deque(maxlen=cfg.kelly_lookback) for b in sds}
    trades: list[dict] = []
    eq_rec: list[dict] = []

    equity = cfg.initial_equity         # 已实现净值 (含入场费扣减)
    fee_paid_total = 0.0
    stats = dict(signals=0, orders_placed=0, orders_filled=0, blocked_slots=0,
                 blocked_gross=0, blocked_kelly=0, corr_halved=0, extensions=0,
                 skipped_no_next_bar=0, late_fill_lookahead=0)

    def gross_used() -> float:
        g = sum(p.size_frac for p in open_pos.values())
        g += sum(o["size_frac"] for o in pending.values())
        return g

    def close_position(b: str, p: Position, exit_ts: int, exit_price: float,
                       reason: str):
        nonlocal equity, fee_paid_total
        gross = exit_price / p.entry_price - 1.0
        fee_rt = 2.0 * cfg.fee_maker
        net = gross - fee_rt
        pnl = p.notional * net
        equity += pnl            # 离场净入 (入场费已在成交时扣)
        fee_paid_total += p.notional * fee_rt
        trades.append(dict(
            base=b, side="long",
            signal_time=pd.Timestamp(p.signal_ts, unit="ns", tz="UTC"),
            entry_time=pd.Timestamp(p.entry_ts, unit="ns", tz="UTC"),
            exit_time=pd.Timestamp(exit_ts, unit="ns", tz="UTC"),
            entry_price=p.entry_price, exit_price=exit_price,
            size_frac=p.size_frac, notional=p.notional,
            bars_held=int((exit_ts - p.entry_ts) // BAR_NS),
            extensions=p.extensions, signal_strength=p.signal_strength,
            pred_ret_raw=p.pred_raw, confidence=p.conf,
            exit_reason=reason, gross_ret=gross, net_ret=net, pnl_equity=pnl,
            fee_equity=p.notional * fee_rt,
        ))
        trade_hist[b].append(net)
        del open_pos[b]

    m = len(ev_ts)
    i = 0
    for t in timeline.tolist():
        # ---- 收集本 bar 的 (base, row) 事件 ----
        ev: dict[str, int] = {}
        while i < m and ev_ts[i] == t:
            bi = int(ev_base[i])
            b = base_list[bi]
            ev[b] = int(ev_row[i])
            ptrs[b] = int(ev_row[i]) + 1
            i += 1

        # ---- 0) 刷新持仓最新价 (供盯市) ----
        for b, r in ev.items():
            p = open_pos.get(b)
            if p is not None:
                p.last_price = sds[b].close[r]

        # ---- 1) 挂单成交判定 (限价单只对下一根 bar 有效) ----
        for b in list(pending.keys()):
            o = pending[b]
            if o["check_ts"] != t or b not in ev:
                continue
            r = ev[b]
            sd = sds[b]
            if sd.low[r] < o["limit"]:
                fill_price = min(o["limit"], sd.open[r])   # 跳空低开按更优的开盘价
                notional = o["size_frac"] * equity
                open_pos[b] = Position(
                    base=b, entry_ts=t, entry_row=r, entry_price=fill_price,
                    size_frac=o["size_frac"], notional=notional,
                    signal_ts=o["signal_ts"], signal_strength=o["signal_strength"],
                    pred_raw=o["pred_raw"], conf=o["conf"],
                    exit_ts=t + cfg.hold_init_bars * BAR_NS,
                    last_price=fill_price)
                equity -= notional * cfg.fee_maker        # 入场 maker 费
                stats["orders_filled"] += 1
            del pending[b]

        # ---- 2) 离场 / 延期 (到期 bar 的收盘价离场) ----
        for b in list(open_pos.keys()):
            p = open_pos[b]
            if b not in ev or t < p.exit_ts:
                continue
            r = ev[b]
            sd = sds[b]
            held_bars = (t - p.entry_ts) // BAR_NS
            same_dir = bool(sd.sig_mask[r])
            if same_dir and held_bars + cfg.hold_extend_bars <= cfg.hold_max_bars:
                p.exit_ts = t + cfg.hold_extend_bars * BAR_NS
                p.extensions += 1
                stats["extensions"] += 1
            else:
                reason = "max_hold" if held_bars >= cfg.hold_max_bars else "signal_off"
                close_position(b, p, t, sd.close[r], reason)

        # ---- 3) 盯市净值 ----
        unreal = sum(p.notional * (p.last_price / p.entry_price - 1.0)
                     for p in open_pos.values())
        eq_rec.append(dict(date=pd.Timestamp(t, unit="ns", tz="UTC"),
                           equity=equity + unreal, realized=equity,
                           unrealized=unreal, gross_exposure=gross_used(),
                           n_positions=len(open_pos)))

        if equity + unreal <= 0:
            print("!! 净值 <= 0, 提前终止回测")
            break

        # ---- 4) 新信号 (本 bar 收盘后挂限价单) ----
        for b, r in ev.items():
            sd = sds[b]
            if not sd.sig_mask[r] or b in open_pos or b in pending:
                continue
            stats["signals"] += 1
            if len(open_pos) + len(pending) >= cfg.max_positions:
                stats["blocked_slots"] += 1
                continue
            if r + 1 >= sd.n:               # 没有下一根 bar, 挂单无处成交
                stats["skipped_no_next_bar"] += 1
                continue

            size = min(volatility_size(sd.vol[r], cfg),
                       kelly_fraction(trade_hist[b]))
            if size <= 0:
                stats["blocked_kelly"] += 1
                continue
            if high_corr_count(b, list(open_pos.keys()), sds, ptrs, t, cfg) \
                    >= cfg.corr_dedup_min:
                size *= cfg.corr_size_factor
                stats["corr_halved"] += 1
            size = min(size, cfg.max_pos_frac, cfg.max_gross - gross_used())
            if size < cfg.min_pos_frac:
                stats["blocked_gross"] += 1
                continue

            limit = sd.close[r] * (1.0 - cfg.entry_limit_offset)
            pending[b] = dict(check_ts=int(sd.ts[r + 1]), limit=limit,
                              size_frac=float(size), signal_ts=t,
                              signal_strength=float(sd.pred[r]),
                              pred_raw=float(sd.pred[r]), conf=float(sd.conf[r]))
            stats["orders_placed"] += 1

    # ---- 数据末尾强制平仓 ----
    for b, p in list(open_pos.items()):
        close_position(b, p, int(sds[b].ts[-1]), p.last_price, "end_of_data")

    stats["fee_paid_total"] = fee_paid_total
    return trades, eq_rec, stats


# --------------------------------------------------------------------------
# 5. 绩效指标与输出
# --------------------------------------------------------------------------
def compute_metrics(eq_rec: list[dict], trades: list[dict],
                    cfg: BacktestConfig) -> dict:
    eq = pd.DataFrame(eq_rec)
    eq_out = eq[["date", "equity", "realized", "unrealized",
                 "gross_exposure", "n_positions"]]

    perf = dict(total_return=None, cagr=None, ann_vol=None, sharpe=None,
                sortino=None, max_drawdown=None, calmar=None, n_bars=len(eq))
    if len(eq) > 1:
        curve = eq.set_index("date")["equity"]
        total = curve.iloc[-1] / curve.iloc[0] - 1.0
        days = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days, 1)
        cagr = (curve.iloc[-1] / curve.iloc[0]) ** (365.0 / days) - 1.0 \
            if curve.iloc[-1] > 0 else -1.0
        runmax = curve.cummax()
        mdd = float((1.0 - curve / runmax).max())
        daily = curve.resample("1D").last().dropna()
        r = daily.pct_change().dropna()
        if len(r) > 2 and r.std() > 0:
            sharpe = float(r.mean() / r.std() * math.sqrt(365.0))
            downside = r[r < 0]
            dstd = math.sqrt(float((downside ** 2).mean())) if len(downside) else 0.0
            sortino = float(r.mean() / dstd * math.sqrt(365.0)) if dstd > 0 else None
            ann_vol = float(r.std() * math.sqrt(365.0))
        else:
            sharpe = sortino = ann_vol = None
        perf.update(total_return=float(total), cagr=float(cagr), ann_vol=ann_vol,
                    sharpe=sharpe, sortino=sortino, max_drawdown=mdd,
                    calmar=float(cagr / mdd) if (mdd and mdd > 0 and cagr is not None) else None)

    tr = dict(n_trades=len(trades))
    if trades:
        tdf = pd.DataFrame(trades)
        net = tdf["net_ret"]
        wins, losses = net[net > 0], net[net < 0]
        expectancy = float(net.mean())
        std = float(net.std(ddof=1)) if len(net) > 1 else None
        tstat = float(expectancy / (std / math.sqrt(len(net)))) if std and std > 0 else None
        avg_gross_exp = float(pd.DataFrame(eq_rec)["gross_exposure"].mean())
        tr.update(
            win_rate=float(len(wins) / len(net)),
            avg_net_ret_bps=expectancy * 1e4,
            payoff_ratio=float(wins.mean() / abs(losses.mean()))
            if len(wins) and len(losses) else None,
            profit_factor=float(wins.sum() / abs(losses.sum()))
            if len(losses) and losses.sum() != 0 else None,
            avg_hold_hours=float(tdf["bars_held"].mean() * BAR_HOURS),
            avg_gross_exposure=avg_gross_exp,
            total_fee_equity=float(tdf["fee_equity"].sum()),
            # 核心问题: 扣除执行摩擦后是否仍有正期望
            expectancy_after_fees_bps=expectancy * 1e4,
            t_stat=tstat,
            positive_expectancy=bool(expectancy > 0),
        )
    return dict(performance=perf, trades=tr)


TRADE_COLS = ["base", "side", "signal_time", "entry_time", "exit_time",
              "entry_price", "exit_price", "size_frac", "notional", "bars_held",
              "extensions", "signal_strength", "pred_ret_raw", "confidence",
              "exit_reason", "gross_ret", "net_ret", "pnl_equity", "fee_equity"]


def write_outputs(out_dir: Path, suffix: str, trades: list[dict],
                  eq_rec: list[dict], report: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    if trades:
        tdf = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    else:
        tdf = pd.DataFrame({c: pd.Series(dtype="float64") for c in TRADE_COLS})
    tdf.to_feather(out_dir / f"backtest_trades{suffix}.feather")
    pd.DataFrame(eq_rec).to_feather(out_dir / f"backtest_equity{suffix}.feather")
    (out_dir / f"backtest_report{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


# --------------------------------------------------------------------------
# mock 数据 (predictions 缺失时验证引擎逻辑)
# --------------------------------------------------------------------------
def make_mock_predictions(klines: dict[str, pd.DataFrame], start: str, end: str,
                          alpha: float = 0.5, noise_mult: float = 0.6,
                          horizon_bars: int = 4, seed: int = 42) -> pd.DataFrame:
    """用真实 K 线合成带 alpha 的模拟预测: pred = alpha*前瞻收益 + 噪声"""
    rng = np.random.default_rng(seed)
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC")
    parts = []
    for b, df in klines.items():
        d = df["date"]
        m = (d >= lo) & (d <= hi)
        if m.sum() < 600:
            continue
        c = df["close"].to_numpy()
        fwd = np.full(len(c), np.nan)
        fwd[:-horizon_bars] = c[horizon_bars:] / c[:-horizon_bars] - 1.0
        sig = pd.Series(fwd).rolling(96, min_periods=48).std().to_numpy()
        noise = rng.normal(0.0, 1.0, len(c)) * sig * noise_mult
        pred = alpha * fwd + noise
        conf = 0.5 + 0.45 * np.clip(np.abs(pred) / (2.0 * sig + 1e-12), 0.0, 1.0)
        parts.append(pd.DataFrame(dict(
            date=d[m].to_numpy(),
            base=b,
            pred_ret=pred[m],
            pred_direction=(pred > 0).astype(int)[m],
            pred_confidence=conf[m],
        )))
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["pred_ret", "pred_confidence"])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ML 策略层回测 (信号+仓位+执行模拟)",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=PRED_PATH,
                   help=f"predictions.feather 路径 (默认 {PRED_PATH})")
    p.add_argument("--out-dir", type=Path, default=ML_DIR)
    p.add_argument("--start", type=str, default=None, help="回测起始日 YYYY-MM-DD (含)")
    p.add_argument("--end", type=str, default=None, help="回测结束日 YYYY-MM-DD (含)")
    p.add_argument("--bases", type=str, default=None,
                   help="逗号分隔 base 白名单, 如 BTC,ETH")
    p.add_argument("--threshold", type=float, default=0.001)
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--ema-span", type=int, default=4)
    p.add_argument("--max-positions", type=int, default=6)
    p.add_argument("--target-vol", type=float, default=0.01)
    p.add_argument("--mock", action="store_true",
                   help="无真实 predictions 时用模拟信号验证引擎 (输出 *_mock 文件)")
    p.add_argument("--mock-alpha", type=float, default=0.5, help="模拟信号的信息系数")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    args = parse_args()
    only = {b.strip().upper() for b in args.bases.split(",")} if args.bases else None

    if args.mock:
        print("[mock] 用真实 K 线 + 合成信号做冒烟测试")
        bases = discover_bases(None, only)
        klines = load_klines(bases, args.start or "2025-01-01", args.end or "2025-02-01")
        pred_df = make_mock_predictions(klines, args.start or "2025-01-01",
                                        args.end or "2025-02-01",
                                        alpha=args.mock_alpha, seed=args.seed)
        mode, suffix = "mock", "_mock"
    else:
        if not args.predictions.exists():
            print(f"[错误] 找不到 {args.predictions}")
            print("       模型层尚未产出 predictions.feather。可先运行:")
            print(f"         {Path(sys.executable).name} scripts/ml/strategy_backtest.py --mock")
            sys.exit(2)
        pred_df = pd.read_feather(args.predictions)
        missing = [c for c in REQUIRED_PRED_COLS if c not in pred_df.columns]
        if missing:
            print(f"[错误] predictions 缺少列: {missing}, 需要 {list(REQUIRED_PRED_COLS)}")
            sys.exit(2)
        if args.start:
            pred_df = pred_df[pd.to_datetime(pred_df["date"], utc=True)
                              >= pd.Timestamp(args.start, tz="UTC")]
        if args.end:
            pred_df = pred_df[pd.to_datetime(pred_df["date"], utc=True)
                              <= pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)]
        bases = discover_bases(pred_df, only)
        klines = load_klines(bases, args.start, args.end)
        mode, suffix = "real", ""
        dropped = set(pred_df["base"].unique()) - set(bases)
        if dropped:
            print(f"[警告] 以下 base 无 15m K 线, 已跳过: {sorted(dropped)}")

    if not bases:
        print("[错误] 没有可用的 base (K 线缺失或白名单为空)")
        sys.exit(2)

    cfg = BacktestConfig(ema_span=args.ema_span, threshold=args.threshold,
                         min_confidence=args.min_confidence,
                         max_positions=args.max_positions,
                         target_vol=args.target_vol)

    print(f"载入 {len(bases)} 个 base, {len(pred_df)} 行预测, "
          f"K 线 {sum(len(v) for v in klines.values())} bar")
    sds = build_series_data(klines, pred_df, cfg)

    trades, eq_rec, stats = run_backtest(sds, cfg)
    metrics = compute_metrics(eq_rec, trades, cfg)

    report = dict(
        mode=mode,
        generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
        period=dict(start=str(pd.DataFrame(eq_rec)["date"].iloc[0]) if eq_rec else None,
                    end=str(pd.DataFrame(eq_rec)["date"].iloc[-1]) if eq_rec else None),
        n_bases=len(bases), bases=bases,
        config=asdict(cfg),
        execution=stats,
        **metrics,
        files=dict(trades=f"backtest_trades{suffix}.feather",
                   equity=f"backtest_equity{suffix}.feather"),
    )
    write_outputs(args.out_dir, suffix, trades, eq_rec, report)

    perf, tr = metrics["performance"], metrics["trades"]
    print("\n===== 回测摘要 =====")
    print(f"模式: {mode} | bar 数: {perf.get('n_bars')} | 交易数: {tr.get('n_trades', 0)}")
    if perf.get("total_return") is not None:
        print(f"总收益: {perf['total_return']*100:+.2f}% | 年化: {(perf['cagr'] or 0)*100:+.2f}%"
              f" | 最大回撤: {(perf['max_drawdown'] or 0)*100:.2f}%")
        print(f"Sharpe: {perf['sharpe'] if perf['sharpe'] is not None else 'NA'}"
              f" | Sortino: {perf['sortino'] if perf['sortino'] is not None else 'NA'}")
    if tr.get("n_trades", 0) > 0:
        print(f"胜率: {tr['win_rate']*100:.1f}% | 盈亏比: {tr['payoff_ratio']:.2f}"
              f" | 期望: {tr['avg_net_ret_bps']:+.2f} bps/笔 (扣费后)")
        print(f"t统计: {tr['t_stat'] if tr['t_stat'] is not None else 'NA'}"
              f" | 正期望: {tr['positive_expectancy']}")
    print(f"挂单 {stats['orders_placed']} -> 成交 {stats['orders_filled']}"
          f" (成交率 {stats['orders_filled']/max(stats['orders_placed'],1)*100:.1f}%)"
          f" | 延期 {stats['extensions']} | 相关性降权 {stats['corr_halved']}"
          f" | Kelly拦截 {stats['blocked_kelly']}")
    print(f"输出目录: {args.out_dir}")
    print(f"耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

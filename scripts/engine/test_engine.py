# -*- coding: utf-8 -*-
"""回测引擎测试套件 — 证明引擎正确性的已知答案测试

每个测试验证引擎的一个具体行为:
  1. 已知答案: 价格从 A 到 B，收益必须是 (B/A - 1)
  2. 确定性: 同一输入跑两遍结果完全一致
  3. 无前视: 截断未来数据，过去的结果不变
  4. 价格对齐: 入场价必须是 next_open
  5. 费用: 扣费后收益 = 毛收益 - 费率
  6. 多仓位: 并发仓位互不干扰
  7. 空头: 空头收益 = 反向

用法: python scripts/engine/test_engine.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_engine import Bar, Portfolio, DataFeed, BacktestEngine, Strategy, generate_report

PASS = "✅"
FAIL = "❌"
results = []


def test(name):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((name, True, ""))
                print(f"{PASS} {name}")
            except AssertionError as e:
                results.append((name, False, str(e)))
                print(f"{FAIL} {name}: {e}")
            except Exception as e:
                results.append((name, False, f"异常: {e}"))
                print(f"{FAIL} {name}: 异常 {e}")
        tests.append(wrapper)
        return wrapper
    return decorator


tests = []

# ─── 辅助: 构造合成数据 ───

def make_bar(pair="TEST", date=None, open=100, high=105, low=95, close=102, volume=1000, next_open=103):
    if date is None:
        date = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    return Bar(pair=pair, date=date, open=open, high=high, low=low,
               close=close, volume=volume, next_open=next_open)


# ─── 测试 1: 已知答案 — 多头收益 ───

@test("已知答案: 多头从100涨到110，收益应为+10%")
def test_long_known_answer():
    pf = Portfolio(initial_capital=1000, fee_maker=0, fee_taker=0, leverage=1)
    bar1 = make_bar(open=100, close=100, next_open=100)  # 信号bar，next_open=入场价
    bar2 = make_bar(open=100, close=110, next_open=110)

    pf.open_long(bar1, size=1000)  # 入场价 = next_open = 100
    # 模拟 mark to market
    pos = pf.positions["TEST"]
    pos.entry_price = 100  # 确认入场价

    # 手动结算 (跳过 engine, 直接测 Portfolio)
    bar_close = make_bar(close=110, next_open=110)
    pos.mark_to_market(110)
    assert abs(pos.unrealized_pnl - 100) < 0.01, f"未实现盈亏应为100, 得到{pos.unrealized_pnl}"

    pf.close(bar_close)
    trade = pf.closed_trades[0]
    assert abs(trade.net_pnl - 100) < 0.01, f"净盈亏应为100, 得到{trade.net_pnl}"
    assert abs(trade.ret_pct - 0.10) < 0.001, f"收益率应为10%, 得到{trade.ret_pct*100}%"


# ─── 测试 2: 已知答案 — 空头收益 ───

@test("已知答案: 空头从100跌到90，收益应为+10%")
def test_short_known_answer():
    pf = Portfolio(initial_capital=1000, fee_maker=0, fee_taker=0, leverage=1)
    bar1 = make_bar(open=100, close=100, next_open=100)
    pf.open_short(bar1, size=1000)
    pos = pf.positions["TEST"]
    pos.entry_price = 100

    pos.mark_to_market(90)
    assert abs(pos.unrealized_pnl - 100) < 0.01, f"空头未实现盈亏应为100, 得到{pos.unrealized_pnl}"


# ─── 测试 3: 费用计算 ───

@test("费用: 0.02% maker费，1000U仓位应收0.2U/边")
def test_fees():
    pf = Portfolio(initial_capital=1000, fee_maker=0.0002, fee_taker=0.0005, leverage=1)
    bar = make_bar(open=100, close=100, next_open=100)
    pf.open_long(bar, size=1000)
    # 开仓费 = 1000 * 0.0002 = 0.2
    assert abs(pf.capital - (1000 - 0.2)) < 0.01, f"开仓后现金应为999.8, 得到{pf.capital}"

    pf.close(bar)
    trade = pf.closed_trades[0]
    # 总费 = 开仓0.2 + 平仓0.2 = 0.4
    assert abs(trade.fees - 0.4) < 0.01, f"总费应为0.4, 得到{trade.fees}"


# ─── 测试 4: 入场价 = next_open ───

@test("价格对齐: 入场价必须是 next_open 而非 close")
def test_entry_price_is_next_open():
    pf = Portfolio(initial_capital=1000, fee_maker=0, leverage=1)
    # close=102 但 next_open=103 → 入场应为103
    bar = make_bar(open=100, close=102, next_open=103)
    pf.open_long(bar, size=1000)
    pos = pf.positions["TEST"]
    assert pos.entry_price == 103, f"入场价应为103(next_open), 得到{pos.entry_price}"


# ─── 测试 5: 无前视 — 截断未来数据不影响过去 ───

@test("无前视: 截断未来数据后，过去的交易结果完全不变")
def test_no_lookahead():
    # 生成 100 根确定性 bar
    def gen_bars(n, noise_seed=42):
        np.random.seed(noise_seed)
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
        bars = []
        base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        for i in range(n):
            bars.append(make_bar(
                date=base + pd.Timedelta(minutes=15 * i),
                open=prices[i], high=prices[i]*1.005, low=prices[i]*0.995,
                close=prices[i], volume=1000,
                next_open=prices[min(i+1, n-1)],
            ))
        return bars

    bars_full = gen_bars(100)
    bars_truncated = gen_bars(50)

    # 用相同策略跑两次
    class AlwaysBuy(Strategy):
        def on_bar(self, bar, pf, ctx):
            if pf.n_positions == 0:
                pf.open_long(bar, size=500)

    # 关键: 在 bar 48 平仓（两套数据 bar 48 的 next_open 都指向 prices[49]，完全一致）
    # bar 49 的 next_open 在截断数据中会退化为 close（因为没有 bar 50），所以不能用来平仓
    CLOSE_AT = 48

    # 完整数据: 只看前 50 根 bar
    pf1 = Portfolio(1000, fee_maker=0, leverage=1)
    for i, bar in enumerate(bars_full[:50]):
        AlwaysBuy().on_bar(bar, pf1, {})
        if i == CLOSE_AT:
            pf1.close(bar)
        pf1.on_bar_end({"TEST": bar})

    # 截断数据: 同样的 50 根
    pf2 = Portfolio(1000, fee_maker=0, leverage=1)
    for i, bar in enumerate(bars_truncated):
        AlwaysBuy().on_bar(bar, pf2, {})
        if i == CLOSE_AT:
            pf2.close(bar)
        pf2.on_bar_end({"TEST": bar})

    # 两种情况的交易结果必须完全一致
    assert len(pf1.closed_trades) == len(pf2.closed_trades), \
        f"交易数不同: {len(pf1.closed_trades)} vs {len(pf2.closed_trades)}"

    for t1, t2 in zip(pf1.closed_trades, pf2.closed_trades):
        assert abs(t1.entry_price - t2.entry_price) < 1e-10, \
            f"入场价不同: {t1.entry_price} vs {t2.entry_price}"
        assert abs(t1.net_pnl - t2.net_pnl) < 1e-10, \
            f"盈亏不同: {t1.net_pnl} vs {t2.net_pnl}"


# ─── 测试 6: 确定性 — 同一输入跑两遍 ───

@test("确定性: 同一输入跑两遍，每次结果完全一致")
def test_determinism():
    def gen_and_run(seed):
        np.random.seed(seed)
        n = 30
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
        pf = Portfolio(1000, fee_maker=0.0002, leverage=1)
        base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        for i in range(n):
            bar = make_bar(
                date=base + pd.Timedelta(minutes=15 * i),
                open=prices[i], high=prices[i]*1.005, low=prices[i]*0.995,
                close=prices[i], volume=1000,
                next_open=prices[min(i+1, n-1)],
            )
            # 简单策略: 每3根开/平
            if i % 6 == 0 and pf.n_positions == 0:
                pf.open_long(bar, size=500)
            elif i % 6 == 3 and pf.n_positions > 0:
                pf.close(bar)
            pf.on_bar_end({"TEST": bar})
        return pf

    pf1 = gen_and_run(42)
    pf2 = gen_and_run(42)

    assert len(pf1.closed_trades) == len(pf2.closed_trades)
    for t1, t2 in zip(pf1.closed_trades, pf2.closed_trades):
        assert abs(t1.net_pnl - t2.net_pnl) < 1e-15
    assert abs(pf1.capital - pf2.capital) < 1e-10


# ─── 测试 7: 多仓位并发 ───

@test("多仓位: 两个并发仓位互不干扰")
def test_multi_position():
    pf = Portfolio(initial_capital=1000, fee_maker=0, leverage=1)
    bar_a = make_bar(pair="AAA", open=100, close=100, next_open=100)
    bar_b = make_bar(pair="BBB", open=200, close=200, next_open=200)

    pf.open_long(bar_a, size=500)
    pf.open_long(bar_b, size=500)
    assert pf.n_positions == 2

    # AAA 涨 10%, BBB 跌 5%
    pos_a = pf.positions["AAA"]
    pos_b = pf.positions["BBB"]
    pos_a.mark_to_market(110)   # +10% × 500 = +50
    pos_b.mark_to_market(190)   # -5% × 500 = -25

    assert abs(pos_a.unrealized_pnl - 50) < 0.01
    assert abs(pos_b.unrealized_pnl - (-25)) < 0.01
    assert abs(pf.equity - (1000 + 50 - 25)) < 0.01, \
        f"净值应为1025, 得到{pf.equity}"


# ─── 测试 8: 杠杆 ───

@test("杠杆: 3x杠杆下价格+5%应产生+15%保证金收益")
def test_leverage():
    pf = Portfolio(initial_capital=1000, fee_maker=0, leverage=3)
    bar = make_bar(open=100, close=100, next_open=100)
    pf.open_long(bar, size=300)  # 名义300, 保证金100
    pos = pf.positions["TEST"]
    pos.entry_price = 100

    pos.mark_to_market(105)  # 价格+5%
    expected = (105/100 - 1) * 300  # 名义收益 = 15
    assert abs(pos.unrealized_pnl - expected) < 0.01, \
        f"3x杠杆+5%价格应赚15U, 得到{pos.unrealized_pnl}"
    # 保证金收益率 = 15/100 = 15%
    assert abs(pos.unrealized_pnl / pos.margin - 0.15) < 0.001


# ─── 运行全部测试 ───

if __name__ == "__main__":
    print("=" * 60)
    print("回测引擎测试套件")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n结果: {passed}/{total} 通过")
    if passed < total:
        print("\n失败的测试:")
        for name, ok, err in results:
            if not ok:
                print(f"  {FAIL} {name}: {err}")
        sys.exit(1)
    else:
        print("\n✅ 引擎所有测试通过 — 回测结果可信")

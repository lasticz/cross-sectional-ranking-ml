# Cross-Sectional Ranking ML System

Multi-asset ranking prediction system using LightGBM with walk-forward validation, feature ablation analysis, and production deployment via NautilusTrader.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)]() [![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)]() [![NautilusTrader](https://img.shields.io/badge/Engine-NautilusTrader%201.231-orange)]()

## Problem

Given N assets at time t, predict which assets will outperform the cross-sectional median over the next 4 hours. This is a **relative ranking problem** — the model doesn't predict absolute returns, but rather which assets will rank higher than others.

This formulation is analogous to learning-to-rank in search/ads systems: the model learns *relative ordering* rather than absolute scores, which is both easier to learn and more robust to distribution shift.

## Key Results

| Metric | Value |
|---|---|
| Rank IC (OOS) | 0.028 (IR = 0.108, t-stat = 38.5) |
| D10-D1 Spread | +8.5 bps / 4h |
| Feature Families | 7 (ablation-tested) |
| Selected Features | 50 (from 123 candidates) |
| Walk-Forward Folds | 43 (12mo train → 1mo test) |
| Backtest Period | 3.5 years (2023-01 → 2026-08) |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                            │
│  27 assets × 3.5 years × 15min bars = 4M+ samples       │
│  OHLCV + funding rate + cross-sectional normalization   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   FEATURE LAYER                          │
│  123 candidate features across 7 families:              │
│  Momentum · Reversal · Volatility · Liquidity           │
│  BTC-Relative · Cross-Sectional Rank · Regime           │
│  Each feature: raw + rolling z-score + CS rank          │
│  IC screening → hierarchical clustering → 50 selected   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   MODEL LAYER                            │
│  LightGBM regressor on market-neutral returns           │
│  Walk-Forward CV: 12mo train → 1mo predict → roll       │
│  Purged + embargo (no lookahead leakage)                │
│  Label: y_i = r_i - median(r_all)  (cross-sectional)    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 PORTFOLIO LAYER                          │
│  Every 8h: rank all assets → long Top3, short Bottom3   │
│  Buffer zone: top-5 held assets not rotated out         │
│  Min holding period: 8h                                 │
│  Position sizing: 10% equity per position, 3x leverage  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                EXECUTION LAYER                           │
│  NautilusTrader: same code for backtest & live          │
│  Dual-engine cross-validation (NT vs custom, Δ≈6%)     │
│  Binance Futures adapter (limit + stop-market orders)   │
└─────────────────────────────────────────────────────────┘
```

## Key Methodological Decisions

### 1. Cross-Sectional Label (not absolute return)

**Before**: predict `r_i = P_i[t+16]/P_i[t] - 1` for each asset independently.
**After**: predict `y_i = r_i - median(r_all)`.

This removes market direction (the dominant noise source) and reduces label variance by 37%. The model answers "which asset is stronger?" instead of "will the market go up?" — a fundamentally easier learning problem.

### 2. Feature Ablation (not feature dumping)

Features are added family-by-family, measuring OOS Rank IC incrementally:

| Added Family | Cumulative IC | Increment |
|---|---|---|
| Momentum only | +0.0094 | baseline |
| + Volatility | +0.0153 | +0.0059 |
| + BTC Relative | +0.0176 | +0.0023 |
| All 7 families | +0.0190 | +0.0014 |

This tells us *what drives the signal* — volatility features (realized vol, vol-of-vol) and cross-asset relative strength are the primary alpha sources.

### 3. Dual-Engine Cross-Validation

Two independent implementations (NautilusTrader + custom pandas engine) run the same strategy. Cross-validation gap: ~6%. This catches bugs like a critical leverage miscalculation (margin vs notional confusion) that inflated returns 3×.

### 4. Offline-Online Gap Analysis

| Stage | Annual Return | Gap Source |
|---|---|---|
| Research (close-price entry) | +322% | — |
| Engine (next-bar-open) | +441% | leverage fix |
| freqtrade (per-pair framework) | -96% | no cross-sectional support |
| **Realistic estimate** | **60-130%** | survivorship bias, slippage, market impact |

## Project Structure

```
├── scripts/
│   ├── ml_v2/                  # Current ML pipeline
│   │   ├── 01_panel_builder.py     # Panel data + 3 label types
│   │   ├── 02_feature_families.py  # 7 families × ablation
│   │   ├── 03_model_and_portfolio.py  # WFCV + Top3/Bottom3
│   │   ├── 04_freqtrade_bridge.py  # Signal file generation
│   │   └── 08_save_deploy_model.py # Model serialization
│   ├── engine/                 # Backtest infrastructure
│   │   ├── backtest_engine.py      # Event-driven engine (8/8 tests)
│   │   ├── test_engine.py          # Known-answer test suite
│   │   ├── nt_mr_backtest.py       # NautilusTrader MR (1.28pp gap vs freqtrade)
│   │   ├── nt_xs_ml_backtest.py    # NautilusTrader ML cross-sectional
│   │   └── validate_mr.py          # Engine vs freqtrade comparison
│   ├── onchain/                # On-chain smart money tracking
│   └── crossex/                # Cross-exchange funding arbitrage
├── user_data/strategies/       # 13 strategy implementations
├── deploy/                     # Server deployment (Docker + systemd)
└── RESEARCH.md                 # Full research log (30+ experiments)
```

## Tech Stack

- **ML**: LightGBM, scikit-learn, pandas, NumPy
- **Backtest**: NautilusTrader 1.231 (Rust core), custom event-driven engine
- **Data**: Binance Futures 15m OHLCV, funding rates, on-chain (Ethereum)
- **Deployment**: Docker, systemd, Debian Linux server
- **Validation**: 8-test unit suite, dual-engine cross-check, freqtrade comparison

## Lessons Learned

1. **Label design is the highest-leverage decision** — switching from absolute to cross-sectional labels doubled IC IR without changing the model.
2. **Offline-online gap is real and large** — theoretical returns shrink 50-70% after modeling execution friction. Only validated through dry-run.
3. **Feature ablation > feature dumping** — understanding *why* the model works enables principled improvement, not blind parameter tuning.
4. **Selection bias matters enormously** — using today's top-27 assets to backtest 2023 inflates returns by 30-50%. Dynamic universe construction is essential.
5. **Dual-engine validation catches critical bugs** — the margin/notional confusion (3× return inflation) was only discovered by comparing two independent implementations.

---

*This project was developed with AI assistance (Claude for implementation, debugging, and API exploration; human for strategy design, validation criteria, and architectural decisions).*

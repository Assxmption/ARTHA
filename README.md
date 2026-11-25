<div align="center">
  <h1>ARTHA</h1>
  <p><strong>Indian Markets (NSE) & Commodities (MCX) Intelligence System</strong></p>
  <p>
    <img alt="Tests" src="https://img.shields.io/badge/tests-752%20passed-brightgreen" />
    <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue" />
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green" />
  </p>
</div>

---

## 🎯 What ARTHA Is

ARTHA is a production-grade, multi-agent financial intelligence system focused on Indian equities (NSE) and commodities (MCX). It bridges three distinct worlds:

1. **Structured Fundamentals Analysis** — Row-level rigorous data, in the style of [Screener.in](https://www.screener.in/).
2. **Domain-Specific AI Narration** — Expert, auditable narrative synthesis over structured data, modeled after architectures like [Harvey AI](https://www.harvey.ai/).
3. **Deterministic Quantitative Engine** — Statistical arbitrage, Hidden Markov Model (HMM) regime detection, and RL-conditioned position sizing, inspired by the publicly-documented methodologies of Renaissance Technologies (Medallion) and the TradingAgents framework.

**Core invariant:** LLMs in ARTHA *never* compute or assert a financial number. All numerical analysis is handled by a deterministic Python quant engine. The AI agents read validated outputs and construct narratives, ensuring 100% traceability and zero numeric hallucinations.

> [!IMPORTANT]
> This is a research/educational system. Generated reports describe validated signals and their historical backtest performance; they do not tell a user what to buy or sell.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARTHA Architecture                           │
├────────────────────┬────────────────────┬───────────────────────────┤
│    Data Layer      │   Quant Engine     │   Agent Layer             │
├────────────────────┼────────────────────┼───────────────────────────┤
│ NSE OHLCV + FnO    │ HMM Regime (3-st)  │ Orchestrator (AnalysisCrew)│
│ MCX Bhavcopy       │ Stat-Arb (Kalman)  │ Fundamentals Agent        │
│ Security Master    │ PCA Stat-Arb       │ Quant Narrator Agent      │
│ MCX Contracts      │ Factor Model       │ Sentiment Agent (FinBERT) │
│ Fundamental History│ 40+ Alpha Signals  │ Risk Agent                │
│ FinBERT News       │ Walk-Forward BT    │ Portfolio Writer           │
│ VIX / Macro        │ RL Allocator (DQN) │                           │
│                    │ Paper Trading      │                           │
├────────────────────┴────────────────────┴───────────────────────────┤
│                       Fact Store (Pydantic + SQLite)                │
│         Typed state: Fundamentals | QuantSignals | RiskFlags        │
│         Rule 10: Every number cites its source                     │
├─────────────────────────────────────────────────────────────────────┤
│                   LiteLLM Tiered Router                             │
│    Cheap (8B) → Mid → Deep (70B+) — with failover & caching       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design decisions** (enforced by [AGENTS.md](AGENTS.md)):

| Rule | Summary |
|------|---------|
| 1 | LLMs never compute numbers — all quant is deterministic code |
| 2 | No sequential prompt-stacking — agents read/write typed Fact Store objects |
| 3 | Tiered model routing — cheap models for retrieval, deep for synthesis |
| 5 | Per-stage checkpointing — failure at stage N resumes from N |
| 6 | Nothing reaches a report unvalidated — signals need walk-forward OOS evidence |
| 7 | Outlier handling: bad data (MAD) ≠ genuine moves (kept for regime detection) |
| 8 | Optimize Sharpe/Sortino, never hardcode return targets |
| 9 | No autonomous live orders without human confirmation |
| 10 | Full traceability — every number cites a Fact Store entry |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)

### Setup
```bash
# Clone
git clone https://github.com/Assxmption/ARTHA.git && cd ARTHA

# Virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Dependencies
pip install -r requirements.txt

# API keys
cp .env.example .env   # Edit with your Groq/Serper keys
```

### Run the Demo
```bash
python scripts/demo_artha.py
```

This runs all components end-to-end on synthetic data:
- Loads 65+ NSE securities from the security master
- Loads 17 MCX commodity contracts
- Trains an HMM regime detector
- Generates multi-strategy signals
- Shows RL allocator (untrained fallback)
- Demonstrates paper trading readiness
- Writes and reads from the Fact Store

### Run Tests
```bash
python -m pytest tests/ -v --tb=short          # offline, deterministic (live tests skipped)
```

Live-market tests hit real NSE / Yahoo / RSS feeds (no fixtures) and check
that bars are current to the latest NSE session, fall on real session dates,
agree across providers, and carry provenance:

```bash
ARTHA_LIVE=1 python -m pytest tests/live -v                    # real market data
ARTHA_LIVE=1 ARTHA_LIVE_LLM=1 python -m pytest tests/live -v   # + real LLM report
```

### Run the Backend
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run the Frontend
```bash
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

---

## 📂 Repository Structure

```
artha/
├── app/
│   ├── data/                    # Data ingestion & providers
│   │   ├── nse.py               #   NSE OHLCV with outlier flagging
│   │   ├── mcx.py               #   MCX bhavcopy ingestion
│   │   ├── security_master.py   #   Point-in-time NSE universe
│   │   ├── mcx_contracts.py     #   MCX contract metadata registry
│   │   ├── nse_fno.py           #   Real NSE options chain fetcher
│   │   ├── nse_long_history.py  #   Provenance-tracked yfinance history
│   │   ├── fundamental_history.py # Look-ahead-free fundamentals
│   │   └── news.py              #   FinBERT / VADER sentiment
│   ├── quant/                   # Deterministic quant engine
│   │   ├── regime.py            #   HMM 3-state detector (BULL/BEAR/SIDEWAYS)
│   │   ├── statarb.py           #   Cointegration + Kalman filter pairs
│   │   ├── pca_statarb.py       #   PCA-based statistical arbitrage
│   │   ├── signals.py           #   40+ alpha signals with MAD z-scoring
│   │   ├── factors.py           #   Cross-sectional factor model
│   │   ├── live_strategy.py     #   Multi-strategy engine (+ RL integration)
│   │   ├── rl_allocator.py      #   DQN regime-conditioned position sizing
│   │   ├── simulator.py         #   Trading simulator with transaction costs
│   │   ├── backtest.py          #   Walk-forward validation protocol
│   │   ├── paper_trade.py       #   Virtual execution daemon (SQLite)
│   │   ├── sleeve_validation.py #   Causal sleeve matrix comparison
│   │   ├── signal_injector.py   #   Validation evidence gate
│   │   └── historical_validation.py # Real-data train/test protocol
│   ├── agents/                  # Multi-agent LLM crew
│   │   ├── orchestrator.py      #   → Redirects to AnalysisCrew
│   │   ├── fundamentals_agent.py
│   │   ├── quant_narrator_agent.py
│   │   ├── sentiment_agent.py
│   │   ├── risk_agent.py
│   │   └── writer_agent.py
│   ├── crew/
│   │   └── analysis_crew.py     #   Direct Python orchestration (no CrewAI stacking)
│   ├── factstore/               # Typed shared state
│   │   ├── schemas.py           #   Pydantic models (FactType, QuantSignal, etc.)
│   │   └── store.py             #   SQLite-backed fact persistence
│   ├── llm/
│   │   └── router.py            #   Tiered LiteLLM config (cheap→mid→deep)
│   └── api/                     # FastAPI endpoints
├── scripts/
│   ├── demo_artha.py            #   End-to-end demo
│   ├── run_multi_strategy_backtest.py  # 9-strategy backtest
│   ├── run_all_sleeve_validations.py   # Expanded sleeve validation
│   ├── run_10yr_oos_validation.py
│   ├── run_real_sleeve_matrix.py
│   └── run_repeated_sleeve_validation.py
├── tests/                       # 752 tests
├── data_cache/
│   ├── nse/security_master.json #   65+ securities with listing dates
│   └── mcx/contracts.json       #   17 commodity futures contracts
├── docs/
│   ├── ARTHA_ARCHITECTURE.md    #   Full system specification
│   ├── ARTHA_IMPLEMENTATION_PLAN.md
│   ├── ARTHA_OPTIMIZATION_ROADMAP.md
│   └── backtest_reports/        #   40+ JSON backtest artifacts
├── frontend/                    # React + Vite UI
└── AGENTS.md                    # Standing contract for all agents
```

---

## 📊 Quantitative Validation Status

The optimization backlog, experiment protocol, and promotion rules are in [`docs/ARTHA_OPTIMIZATION_ROADMAP.md`](docs/ARTHA_OPTIMIZATION_ROADMAP.md).

**Strategy sleeves under validation:**

| Sleeve | Type | Status |
|--------|------|--------|
| Momentum 12m skip-1m | Long-only | Diagnostic (OOS Sharpe varies by fold) |
| Momentum 6m | Long-only | Diagnostic |
| Reversal 1m / 3m | Contrarian | Diagnostic |
| Factor L/S | Long/Short | Diagnostic |
| Mean Reversion 20d | Contrarian | New — added to expanded config |
| Trend Following 200d | Trend | New — added to expanded config |
| Stat-Arb (Kalman) | Pairs | Disabled (insufficient pairs) |
| PCA Stat-Arb | Multi-factor | Active |
| Options Vol-Carry | Iron Condor | Active |
| ML Alpha | Ensemble | Active |

ARTHA treats Sharpe > 1.3 and ~25% CAGR as *stretch objectives*, not backtest targets. Signals are promoted only after real-data, out-of-sample validation with costs, drawdown, capacity, and provenance checks.

---

## 📚 References & Inspiration

| Source | Contribution |
|--------|-------------|
| **TradingAgents** (Xiao et al., 2024) | Agent-role taxonomy, fast/deep model tiering |
| **Avellaneda & Lee (2010)** | PCA-based statistical arbitrage |
| **Moody & Saffell (2001)** | Differential Sharpe ratio for RL reward |
| **Chain-of-Alpha / AlphaAgent** | LLM proposes factors, deterministic engine validates |
| **Harvey AI** | Narration-over-structured-data architecture |
| **Screener.in** | Row-level Indian fundamentals analysis model |

---

## ⚖️ Disclaimer

This system is for research and educational purposes only. It does not constitute investment advice. Users are solely responsible for any decisions made based on the system's outputs. See [AGENTS.md](AGENTS.md) Rule 11.

<div align="center">
  <h1>ARTHA</h1>
  <p><strong>Indian Markets (NSE) & Commodities (MCX) Intelligence System</strong></p>
</div>

---

## 🎯 Our Goal
ARTHA is a production-grade, multi-agent financial intelligence system focused strictly on Indian equities (NSE) and commodities (MCX). 

We are building a system that bridges three distinct worlds:
1. **Structured Fundamentals Analysis:** Row-level rigorous data, in the style of [Screener.in](https://www.screener.in/).
2. **Domain-Specific AI Narration:** Expert, auditable narrative synthesis over structured data, modeled after architectures like [Harvey AI](https://www.harvey.ai/).
3. **Deterministic Quantitative Engine:** Statistical arbitrage and Hidden Markov Model (HMM) regime detection, inspired by the publicly-documented methodologies of Renaissance Technologies (Medallion).

**Crucially:** LLMs in ARTHA *never* compute or assert a financial number. All numerical analysis is handled by a deterministic Python quant engine. The AI agents simply read the validated outputs and construct narratives, ensuring 100% traceability and zero numeric hallucinations.

---

## 🏗️ System Architecture

Our system is broken down into four decoupled layers, addressing the flaws of naive multi-agent stacking (like token bloat and rate limits):

1. **Data Layer (Cached & Bounded):** 
   * NSE historical/EOD data (via `jugaad-data`, `yfinance`).
   * MCX free EOD Bhavcopy.
   * Scoped news feeds and macro data.
2. **Quant Engine (Deterministic):**
   * **Regime Detector:** HMM/Baum-Welch algorithm to identify market regimes.
   * **Stat-Arb Screener:** Cointegration testing and Ornstein-Uhlenbeck (OU) spread fitting for pairs/baskets.
   * **Walk-forward Backtester:** Vectorized backtesting for realistic edge validation.
3. **Fact Store:** 
   * A shared state of typed Pydantic models (Fundamental Rows, Quant Signals, Risk Flags). Agents read from and write to this store instead of passing growing chat transcripts.
4. **Multi-Agent Crew (Tiered LLM Routing):**
   * **Orchestrator & Row Agents:** Fast, cheap models (e.g., 8B class) for data retrieval and fundamental screening.
   * **Risk & Compliance Agent:** Mid-tier models for position sizing and safety checks.
   * **Portfolio Writer:** Deep models (70B+) used *only once* for the final synthesis and narrative reporting.

---

## 🚀 What We Have Achieved So Far

- **Premium UI Foundation:** A fully responsive, dark-mode frontend built with React and Vite. Features include candlestick charting, a global search interface, and live streaming of agent actions via Server-Sent Events (SSE).
- **Backend Architecture:** A FastAPI backend providing the robust API layer and streaming services.
- **Quant Engine Skeleton:** The `app/quant/` directory is seeded with the initial components for backtesting, factor generation, and HMM regime detection.
- **Agent Pipelines:** The fundamental crew and orchestrator logic are in place (`app/agents/`), built around a robust `FactStore` to prevent token limits and context bloat.
- **Version Control & Security:** 
  - Centralized repository initialized on GitHub.
  - Strict branch architectures: `main` (stable releases) and `develop` (active development).
  - Enforced protections against force-pushes and deletions to safeguard commit history.

---

## 🛠️ Replication & Setup (For Teammates)

Welcome to the team! Here is how to get the project running locally.

### 1. Prerequisites
Ensure you have Python 3.11+ and Node.js installed.

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/Assxmption/ARTHA.git
cd ARTHA

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Setup API Keys
cp .env.example .env
```
*Note: Update `.env` with your Groq and Serper API keys. Our tiered rate-limit routing utilizes Groq's free tiers optimally.*

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Running the Application
From the root directory, start the FastAPI server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Navigate to the port provided by Vite (usually `http://localhost:5173`) to view the interactive UI.

---

## 📚 References & Inspiration
Our architecture is heavily grounded in recent literature and top projects:
- **TradingAgents** (Xiao et al., 2024): For the agent-role taxonomy and fast/deep model tiering.
- **Avellaneda & Lee (2010):** The mathematical backbone for PCA-based statistical arbitrage.
- **Chain-of-Alpha / AlphaAgent:** For the loop where an LLM proposes factors but a deterministic engine backtests and gates them.

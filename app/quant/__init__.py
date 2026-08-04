"""
Quantitative Signal Engine
===========================
Deterministic numerical computation for ARTHA.

All numbers in ARTHA originate here — LLM agents narrate these outputs
but never compute or assert their own numbers (AGENTS.md rule 1).

Modules:
  regime.py     — Hidden Markov Model regime detection
  statarb.py    — Cointegration-based statistical arbitrage
  factors.py    — Multi-factor cross-sectional alpha
  backtest.py   — Walk-forward backtester (validation gate)
  alpha_loop.py — Signal orchestrator → Fact Store
"""

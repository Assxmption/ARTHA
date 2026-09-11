# Risk Agent

**Status**: Implemented
**Path**: `app/agents/risk_agent.py`

The Risk Agent is the compliance and safety gatekeeper of the system. It reads all facts (Fundamentals, Quant) and generates `RiskFlag` objects.

## Veto Power (BLOCK Severity)
The Risk Agent possesses hard veto power over the final report. If it issues a `RiskFlag` with `severity=Severity.BLOCK`, that symbol or signal is strictly excluded from the final portfolio recommendation.

## Deterministic Risk Rules
The rules are hardcoded and do not rely on LLM inference to trigger:
- **Sector Concentration**: Flags if >30% exposure in one sector.
- **Drawdown**: Warnings at -10%, Blocks at -20% OOS max drawdown.
- **Sharpe**: Warnings at < 0.5, Blocks at < 0.0 (destroying value).
- **Leverage (D/E)**: Warnings at > 2.0, Blocks at > 5.0.
- **Volatility**: Warnings at > 50%, Blocks at > 80% annualized.
- **Bear Regime**: Emits warnings if the HMM detects a BEAR regime.
- **SEBI Compliance**: Always emits an INFO flag reminding the user of the SEBI retail algo-trading framework (No automated live trading without human-in-the-loop).

## Narration
Uses the **Mid-Tier LLM** to write a concise, actionable 2-3 paragraph summary of the flagged risks, explicitly noting vetoed signals. Has a deterministic fallback if the LLM fails.

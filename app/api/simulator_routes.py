"""
Simulator API Routes — Trading Simulation Endpoints
=====================================================
REST API for the ARTHA automated trading simulator.

Endpoints:
  POST /api/sim/start          — Start a new simulation
  GET  /api/sim/{id}/status    — Current NAV, P&L, metrics
  GET  /api/sim/{id}/trades    — Full trade log
  GET  /api/sim/{id}/positions — Open positions
  GET  /api/sim/{id}/equity    — Equity curve for charting
  GET  /api/sim/{id}/signals   — Current day's signals
  GET  /api/sims               — List all simulations

These are INDEPENDENT of the fundamental analysis pipeline.
The quant engine is a separate automated worker.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("artha.api.simulator")
router = APIRouter(prefix="/api/sim", tags=["Simulator"])

# ── In-memory store for running/completed simulations ────────────────────────

_simulations: dict[str, dict] = {}
_sim_lock = threading.Lock()


# ── Request/Response Models ─────────────────────────────────────────────────

class SimStartRequest(BaseModel):
    capital: float = Field(default=15_00_000.0, description="Starting capital in INR")
    days: int = Field(default=365, description="Number of historical days to replay")
    symbols: list[str] = Field(
        default=[
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "HCLTECH", "AXISBANK", "ASIANPAINT", "MARUTI",
            "SUNPHARMA", "TITAN", "ULTRACEMCO", "BAJFINANCE", "WIPRO",
            "NESTLEIND", "TATAMOTORS", "POWERGRID", "NTPC", "ADANIENT",
        ],
        description="NSE symbols to trade"
    )
    strategy_type: str = Field(default="EQUITY_LONG_ONLY", description="Strategy to run")


# ── Background simulation runner ────────────────────────────────────────────

def _run_simulation(sim_id: str, capital: float, days: int, symbols: list[str], strategy_type: str = "EQUITY_LONG_ONLY"):
    """Run simulation in background thread."""
    import numpy as np
    import pandas as pd
    import yfinance as yf
    
    try:
        with _sim_lock:
            _simulations[sim_id]["status"] = "running"
            _simulations[sim_id]["progress"] = "Downloading market data..."
        
        # Download price data
        tickers = [f"{s}.NS" for s in symbols]
        data = yf.download(tickers, period=f"{days}d", progress=False)
        
        if data.empty:
            with _sim_lock:
                _simulations[sim_id]["status"] = "error"
                _simulations[sim_id]["error"] = "Failed to download market data"
            return
        
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
            prices.columns = [c.replace('.NS', '') for c in prices.columns]
        else:
            prices = data[['Close']]
            prices.columns = [symbols[0]]
        
        prices.index = prices.index.tz_localize(None) if prices.index.tz else prices.index
        
        # Drop columns with >50% NaN
        valid_cols = prices.columns[prices.notna().sum() > len(prices) * 0.5]
        prices = prices[valid_cols]
        
        with _sim_lock:
            _simulations[sim_id]["progress"] = f"Data loaded: {len(prices)} days, {len(prices.columns)} stocks"
        
        # Benchmark
        bench = yf.Ticker("^NSEI").history(period=f"{days}d")
        bench_prices = bench['Close']
        bench_prices.index = bench_prices.index.tz_localize(None)
        
        # Regime detection
        with _sim_lock:
            _simulations[sim_id]["progress"] = "Running HMM regime detection..."
        
        from app.quant.regime import RegimeDetector
        detector = RegimeDetector()
        detector.fit(bench_prices)
        regime_series = detector.predict(bench_prices)
        regime_stats = detector.get_state_statistics()
        current_regime = detector.current_regime(bench_prices)
        
        with _sim_lock:
            _simulations[sim_id]["progress"] = f"Regime: {current_regime}. Running simulation..."
        
        # Run simulation
        if strategy_type == "IRON_CONDOR":
            from app.quant.options_backtest import OptionsBacktester, precompute_signals
            from app.quant.options_strategies import StrategyType
            from app.data.vix import get_vix_close
            
            with _sim_lock:
                _simulations[sim_id]["progress"] = "Pre-computing options signals..."
                
            vix_series = get_vix_close()
            # We assume bench_prices is NIFTY for Iron Condor
            precompute_signals(bench_prices, vix_series, regime_series)
            
            sim = OptionsBacktester(capital=capital, options_pct=0.30, max_concurrent=5)
            result = sim.backtest_strategy(
                close_prices=bench_prices,
                regime_series=regime_series,
                vix_series=vix_series,
                strategy_type=StrategyType.IRON_CONDOR,
                symbol="NIFTY",
                entry_interval=7
            )
        elif strategy_type == "MULTI_STRATEGY":
            import subprocess
            import json
            from pathlib import Path
            
            with _sim_lock:
                _simulations[sim_id]["progress"] = "Starting Multi-Strategy Engine..."
                
            # Run the actual script
            process = subprocess.Popen(
                ["python", "scripts/run_multi_strategy_backtest.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/Users/goral/Documents/multi_agent_stock_analyzer"
            )
            
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    # Clean up the logger prefix to just show the message
                    if "] artha.multi_strategy_backtest:" in line:
                        msg = line.split("] artha.multi_strategy_backtest:")[-1].strip()
                    else:
                        msg = line
                        
                    with _sim_lock:
                        _simulations[sim_id]["progress"] = msg
                        
            process.stdout.close()
            retcode = process.wait()
            
            if retcode != 0:
                raise Exception(f"Multi-strategy script failed with code {retcode}")
                
            # Read the latest report
            reports_dir = Path("docs/backtest_reports")
            reports = sorted(reports_dir.glob("multi_strategy_*.json"))
            if not reports:
                raise Exception("No backtest report found after running script")
                
            latest_report = reports[-1]
            with open(latest_report, "r") as f:
                report_data = json.load(f)
                
            # Prefer hedged_portfolio over portfolio
            port_key = "hedged_portfolio" if "hedged_portfolio" in report_data.get("strategies", {}) else "portfolio"
            metrics = report_data["strategies"].get(port_key, report_data.get("portfolio", {}))
            
            def parse_pct(val):
                if isinstance(val, str) and val.endswith("%"):
                    return float(val.strip("%"))
                return float(val) if val is not None else 0.0
                
            total_ret_pct = parse_pct(metrics.get("total_return", 0))
            
            result_dict = {
                "initial_capital": capital,
                "final_nav": capital * (1 + (total_ret_pct / 100.0)),
                "total_return_pct": total_ret_pct,
                "total_pnl": capital * (total_ret_pct / 100.0),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0)),
                "sortino_ratio": float(metrics.get("sortino_ratio", 0)),
                "max_drawdown_pct": parse_pct(metrics.get("max_drawdown", 0)),
                "win_rate": parse_pct(metrics.get("win_rate", 0)),
                "total_trades": metrics.get("n_trading_days", 0), # Proxy
                "trade_log": [],
            }
        elif strategy_type == "PCA_STATARB":
            from app.quant.pca_statarb import backtest_pca_statarb
            with _sim_lock:
                _simulations[sim_id]["progress"] = "Running PCA StatArb..."
            # Pass all prices to PCA StatArb
            result = backtest_pca_statarb(prices, bench_prices, regime_series, capital=capital)
        else:
            from app.quant.simulator import TradingSimulator
            sim = TradingSimulator(initial_capital=capital)
            result = sim.run_historical_replay(prices, bench_prices, regime_series)
        
        # Store results
        if strategy_type == "IRON_CONDOR":
            result_dict = {
                "initial_capital": capital,
                "final_nav": capital + result.total_pnl,
                "total_return_pct": result.total_return_pct,
                "total_pnl": result.total_pnl,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "total_trades": result.n_trades,
                "trade_log": result.trades,
            }
        elif strategy_type == "MULTI_STRATEGY":
            import subprocess
            import json
            from pathlib import Path
            
            with _sim_lock:
                _simulations[sim_id]["progress"] = "Starting Multi-Strategy Engine..."
                
            # Run the actual script
            process = subprocess.Popen(
                ["python", "scripts/run_multi_strategy_backtest.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/Users/goral/Documents/multi_agent_stock_analyzer"
            )
            
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    # Clean up the logger prefix to just show the message
                    if "] artha.multi_strategy_backtest:" in line:
                        msg = line.split("] artha.multi_strategy_backtest:")[-1].strip()
                    else:
                        msg = line
                        
                    with _sim_lock:
                        _simulations[sim_id]["progress"] = msg
                        
            process.stdout.close()
            retcode = process.wait()
            
            if retcode != 0:
                raise Exception(f"Multi-strategy script failed with code {retcode}")
                
            # Read the latest report
            reports_dir = Path("docs/backtest_reports")
            reports = sorted(reports_dir.glob("multi_strategy_*.json"))
            if not reports:
                raise Exception("No backtest report found after running script")
                
            latest_report = reports[-1]
            with open(latest_report, "r") as f:
                report_data = json.load(f)
                
            # Prefer hedged_portfolio over portfolio
            port_key = "hedged_portfolio" if "hedged_portfolio" in report_data.get("strategies", {}) else "portfolio"
            metrics = report_data["strategies"].get(port_key, report_data.get("portfolio", {}))
            
            def parse_pct(val):
                if isinstance(val, str) and val.endswith("%"):
                    return float(val.strip("%"))
                return float(val) if val is not None else 0.0
                
            total_ret_pct = parse_pct(metrics.get("total_return", 0))
            
            result_dict = {
                "initial_capital": capital,
                "final_nav": capital * (1 + (total_ret_pct / 100.0)),
                "total_return_pct": total_ret_pct,
                "total_pnl": capital * (total_ret_pct / 100.0),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0)),
                "sortino_ratio": float(metrics.get("sortino_ratio", 0)),
                "max_drawdown_pct": parse_pct(metrics.get("max_drawdown", 0)),
                "win_rate": parse_pct(metrics.get("win_rate", 0)),
                "total_trades": metrics.get("n_trading_days", 0), # Proxy
                "trade_log": [],
            }
        elif strategy_type == "PCA_STATARB":
            # PCA StatArb returns pd.Series of daily returns
            total_ret = (1 + result).prod() - 1
            ann_vol = result.std() * np.sqrt(252)
            sharpe = (result.mean() / result.std() * np.sqrt(252)) if result.std() > 0 else 0
            downside = result[result < 0]
            sortino = (result.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0
            cum = (1 + result).cumprod()
            max_dd = (cum / cum.cummax() - 1).min() * 100
            
            result_dict = {
                "initial_capital": capital,
                "final_nav": capital * (1 + total_ret),
                "total_return_pct": total_ret * 100,
                "total_pnl": capital * total_ret,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown_pct": abs(max_dd),
                "win_rate": len(result[result > 0]) / len(result) * 100 if len(result) > 0 else 0,
                "total_trades": len(result[result != 0]), # Proxy
                "trade_log": [{"date": d.strftime("%Y-%m-%d"), "pnl": p} for d, p in result.items() if p != 0],
            }
        else:
            result_dict = vars(result) if hasattr(result, '__dict__') else result
            
        result_dict['regime_stats'] = regime_stats
        result_dict['current_regime'] = str(current_regime)
        result_dict['universe'] = list(prices.columns)
        
        with _sim_lock:
            _simulations[sim_id].update(result_dict)
            _simulations[sim_id]["status"] = "completed"
            _simulations[sim_id]["progress"] = "Simulation complete"
        
        logger.info("Simulation %s completed: %d trades, Sharpe=%.3f",
                     sim_id, result_dict.get('total_trades', 0), 
                     result_dict.get('sharpe_ratio', 0))
    
    except Exception as e:
        logger.error("Simulation %s failed: %s", sim_id, e, exc_info=True)
        with _sim_lock:
            _simulations[sim_id]["status"] = "error"
            _simulations[sim_id]["error"] = str(e)


# ── API Endpoints ───────────────────────────────────────────────────────────

@router.post("/start")
async def start_simulation(req: SimStartRequest, bg: BackgroundTasks):
    """Start a new trading simulation."""
    sim_id = f"sim_{uuid.uuid4().hex[:8]}"
    
    with _sim_lock:
        _simulations[sim_id] = {
            "sim_id": sim_id,
            "status": "pending",
            "initial_capital": req.capital,
            "days": req.days,
            "symbols": req.symbols,
            "progress": "Initializing...",
            "started_at": time.time(),
        }
    
    bg.add_task(_run_simulation, sim_id, req.capital, req.days, req.symbols, req.strategy_type)
    
    return {
        "sim_id": sim_id,
        "status": "pending",
        "message": f"Simulation started with ₹{req.capital:,.0f} capital, {req.days} days, {len(req.symbols)} stocks.",
    }


@router.get("/list")
async def list_simulations():
    """List all simulations."""
    with _sim_lock:
        sims = []
        for sid, s in _simulations.items():
            sims.append({
                "sim_id": sid,
                "status": s.get("status", "unknown"),
                "initial_capital": s.get("initial_capital", 0),
                "final_nav": s.get("final_nav", 0),
                "total_return_pct": s.get("total_return_pct", 0),
                "sharpe_ratio": s.get("sharpe_ratio", 0),
                "total_trades": s.get("total_trades", 0),
                "progress": s.get("progress", ""),
            })
        return {"simulations": sims, "total": len(sims)}


@router.get("/{sim_id}/status")
async def get_simulation_status(sim_id: str):
    """Get simulation status and key metrics."""
    with _sim_lock:
        if sim_id not in _simulations:
            raise HTTPException(404, f"Simulation {sim_id} not found")
        s = _simulations[sim_id].copy()
    
    return {
        "sim_id": sim_id,
        "status": s.get("status"),
        "progress": s.get("progress"),
        "initial_capital": s.get("initial_capital"),
        "final_nav": s.get("final_nav", 0),
        "total_return_pct": s.get("total_return_pct", 0),
        "total_pnl": s.get("total_pnl", 0),
        "sharpe_ratio": s.get("sharpe_ratio", 0),
        "sortino_ratio": s.get("sortino_ratio", 0),
        "max_drawdown_pct": s.get("max_drawdown_pct", 0),
        "win_rate": s.get("win_rate", 0),
        "total_trades": s.get("total_trades", 0),
        "profit_factor": s.get("profit_factor", 0),
        "calmar_ratio": s.get("calmar_ratio", 0),
        "start_date": s.get("start_date", ""),
        "end_date": s.get("end_date", ""),
        "n_days": s.get("n_days", 0),
        "current_regime": s.get("current_regime", "UNKNOWN"),
        "regime_stats": s.get("regime_stats", {}),
        "n_positions": len(s.get("positions", [])),
        "universe": s.get("universe", []),
        "error": s.get("error"),
    }


@router.get("/{sim_id}/trades")
async def get_trades(sim_id: str, limit: int = 100, offset: int = 0):
    """Get the trade log."""
    with _sim_lock:
        if sim_id not in _simulations:
            raise HTTPException(404, f"Simulation {sim_id} not found")
        trades = _simulations[sim_id].get("trade_log", [])
    
    total = len(trades)
    page = trades[offset:offset + limit]
    
    return {
        "sim_id": sim_id,
        "trades": page,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/{sim_id}/positions")
async def get_positions(sim_id: str):
    """Get current open positions."""
    with _sim_lock:
        if sim_id not in _simulations:
            raise HTTPException(404, f"Simulation {sim_id} not found")
        s = _simulations[sim_id]
    
    return {
        "sim_id": sim_id,
        "positions": s.get("positions", []),
        "n_positions": len(s.get("positions", [])),
        "total_invested": sum(p.get("market_value", 0) for p in s.get("positions", [])),
        "total_unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in s.get("positions", [])),
    }


@router.get("/{sim_id}/equity")
async def get_equity_curve(sim_id: str, sample: int = 0):
    """
    Get the equity curve for charting.
    
    Pass sample=N to downsample to N points (useful for large datasets).
    """
    with _sim_lock:
        if sim_id not in _simulations:
            raise HTTPException(404, f"Simulation {sim_id} not found")
        curve = _simulations[sim_id].get("equity_curve", [])
    
    if sample > 0 and len(curve) > sample:
        step = len(curve) // sample
        curve = curve[::step]
    
    return {
        "sim_id": sim_id,
        "equity_curve": curve,
        "n_points": len(curve),
    }


@router.get("/{sim_id}/signals")
async def get_current_signals(sim_id: str):
    """Get the signals from the last trading day."""
    with _sim_lock:
        if sim_id not in _simulations:
            raise HTTPException(404, f"Simulation {sim_id} not found")
        s = _simulations[sim_id]
    
    # Get last day's trades as "signals"
    trades = s.get("trade_log", [])
    if not trades:
        return {"sim_id": sim_id, "signals": [], "date": ""}
    
    last_date = trades[-1].get("date", "")
    last_day_trades = [t for t in trades if t.get("date") == last_date]
    
    signals = []
    for t in last_day_trades:
        signals.append({
            "symbol": t.get("symbol"),
            "action": t.get("action"),
            "quantity": t.get("quantity"),
            "price": t.get("price"),
            "alpha_score": t.get("signal_score"),
            "reason": t.get("reason"),
            "regime": t.get("regime"),
        })
    
    return {
        "sim_id": sim_id,
        "date": last_date,
        "regime": s.get("current_regime", "UNKNOWN"),
        "signals": signals,
    }

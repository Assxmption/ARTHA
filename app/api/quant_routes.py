"""
ARTHA Quant API Routes
======================
REST endpoints for the quant engine dashboard.

Endpoints:
  GET  /api/quant/reports              → List all backtest reports
  GET  /api/quant/reports/latest       → Get the latest report
  GET  /api/quant/reports/{filename}   → Get specific report
  GET  /api/quant/portfolio            → Current portfolio holdings & signals
  GET  /api/quant/performance          → Historical performance data for charting
  GET  /api/quant/regime               → Current and historical regime state
  GET  /api/quant/pairs                → Current pair trading signals
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("artha.api.quant")

quant_router = APIRouter(prefix="/api/quant", tags=["Quant Engine"])

REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "backtest_reports"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


def _load_report(filename: str) -> dict:
    """Load a JSON report from the backtest reports directory."""
    path = REPORT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Report {filename} not found")
    if not path.suffix == ".json":
        raise HTTPException(status_code=400, detail="Invalid report format")
    with open(path) as f:
        return json.load(f)


def _get_latest_report() -> tuple[str, dict]:
    """Get the most recent v4 report, falling back to any report."""
    if not REPORT_DIR.exists():
        raise HTTPException(status_code=404, detail="No reports directory")
    
    reports = sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        raise HTTPException(status_code=404, detail="No reports available")
    
    # Prefer v4 reports
    v4_reports = [r for r in reports if "v4" in r.name]
    target = v4_reports[0] if v4_reports else reports[0]
    
    with open(target) as f:
        return target.name, json.load(f)


@quant_router.get("/reports")
async def list_reports():
    """List all available backtest reports."""
    if not REPORT_DIR.exists():
        return {"reports": []}
    
    reports = []
    for path in sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(path) as f:
            data = json.load(f)
        reports.append({
            "filename": path.name,
            "version": data.get("version", "unknown"),
            "generated_at": data.get("generated_at"),
            "universe_size": data.get("universe_size") or data.get("combined_strategy", {}).get("n_stocks"),
            "benchmark_sharpe": data.get("benchmark_sharpe"),
            "strategy_sharpe": (
                data.get("combined_strategy", {}).get("vt_sharpe")
                or data.get("factor_alpha_tilt", {}).get("vt_sharpe")
            ),
        })
    return {"reports": reports}


@quant_router.get("/reports/latest")
async def get_latest_report():
    """Get the most recent backtest report with full data."""
    filename, data = _get_latest_report()
    return {"filename": filename, **data}


@quant_router.get("/reports/{filename}")
async def get_report(filename: str):
    """Get a specific report by filename."""
    data = _load_report(filename)
    return {"filename": filename, **data}


@quant_router.get("/portfolio")
async def get_portfolio():
    """
    Get current portfolio state: holdings, sector allocation,
    top/bottom picks, and regime-conditional weights.
    """
    filename, report = _get_latest_report()
    strategy = report.get("combined_strategy", {})
    pairs = report.get("pair_trading", {})
    
    return {
        "report_version": report.get("version"),
        "generated_at": report.get("generated_at"),
        "universe_size": report.get("universe_size", strategy.get("n_stocks")),
        "total_sharpe": strategy.get("total_sharpe"),
        "vt_sharpe": strategy.get("vt_sharpe"),
        "total_return": strategy.get("total_return"),
        "vt_return": strategy.get("vt_return"),
        "total_max_dd": strategy.get("total_max_dd"),
        "vt_max_dd": strategy.get("vt_max_dd"),
        "win_rate": strategy.get("total_win_rate"),
        "profit_factor": strategy.get("total_profit_factor"),
        "calmar_ratio": strategy.get("total_calmar"),
        "n_rebalances": strategy.get("n_rebalances"),
        "n_days": strategy.get("n_days"),
        "benchmark_sharpe": report.get("benchmark_sharpe"),
        "regime_performance": strategy.get("regime_performance", {}),
        "pairs_validated": pairs.get("validated", 0),
        "pairs_total": pairs.get("backtested", 0),
        "top_pairs": pairs.get("pairs", [])[:5],
    }


@quant_router.get("/performance")
async def get_performance():
    """
    Get performance comparison data for charting.
    Returns strategy vs benchmark metrics across versions.
    """
    if not REPORT_DIR.exists():
        return {"versions": []}
    
    versions = []
    for path in sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        with open(path) as f:
            data = json.load(f)
        
        version = data.get("version", "unknown")
        strategy = data.get("combined_strategy") or data.get("factor_alpha_tilt", {})
        
        versions.append({
            "version": version,
            "filename": path.name,
            "generated_at": data.get("generated_at"),
            "strategy_sharpe": strategy.get("vt_sharpe") or strategy.get("total_sharpe"),
            "strategy_return": strategy.get("vt_return") or strategy.get("total_return"),
            "strategy_max_dd": strategy.get("vt_max_dd") or strategy.get("total_max_dd"),
            "strategy_vol": strategy.get("total_vol"),
            "benchmark_sharpe": data.get("benchmark_sharpe"),
            "win_rate": strategy.get("total_win_rate") or strategy.get("vt_win_rate"),
            "profit_factor": strategy.get("total_profit_factor"),
        })
    
    return {"versions": versions}


@quant_router.get("/regime")
async def get_regime():
    """
    Get current market regime state from live HMM detection on NIFTY 50.
    
    Runs a 3-state Gaussian HMM on 5 years of NIFTY 50 data to classify
    the current market as BULL, BEAR, or SIDEWAYS. Results are cached
    for 4 hours to avoid repeated computation.
    """
    import time as _time
    
    cache_key = "_regime_cache"
    cache_ttl = 4 * 3600  # 4 hours
    
    # Check cache
    cached = getattr(get_regime, cache_key, None)
    if cached and (_time.time() - cached.get("_ts", 0)) < cache_ttl:
        return cached["data"]
    
    try:
        import yfinance as yf
        from app.quant.regime import RegimeDetector
        
        # Fetch 5 years of NIFTY 50 data
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(period="5y")
        
        if df.empty:
            return {
                "current_regime": "UNKNOWN",
                "regime_statistics": {},
                "regime_timeline": [],
                "error": "Could not fetch NIFTY 50 data",
            }
        
        prices = df['Close']
        prices.index = prices.index.tz_localize(None)
        
        # Run HMM
        detector = RegimeDetector(n_states=3)
        success = detector.fit(prices)
        
        if not success:
            return {
                "current_regime": "UNKNOWN",
                "regime_statistics": {},
                "regime_timeline": [],
                "error": "HMM fitting failed — insufficient data",
            }
        
        current = detector.current_regime(prices)
        stats = detector.get_state_statistics()
        regimes = detector.predict(prices)
        
        # Build regime timeline (transitions)
        transitions = []
        prev = None
        for dt, r in regimes.items():
            label = r.value if hasattr(r, 'value') else str(r)
            if label != "UNKNOWN" and label != prev:
                transitions.append({
                    "date": str(dt.date()),
                    "regime": label,
                })
                prev = label
        
        # Regime distribution over last year
        last_year = regimes[regimes.index >= str((datetime.now().date() - __import__('datetime').timedelta(days=365)))]
        distribution = {}
        for r in last_year:
            label = r.value if hasattr(r, 'value') else str(r)
            distribution[label] = distribution.get(label, 0) + 1
        
        result = {
            "current_regime": current.value if hasattr(current, 'value') else str(current),
            "data_points": len(prices),
            "date_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
            "regime_statistics": stats,
            "regime_transitions": transitions[-20:],  # Last 20 transitions
            "last_year_distribution": distribution,
        }
        
        # Cache
        setattr(get_regime, cache_key, {"data": result, "_ts": _time.time()})
        return result
    
    except Exception as e:
        logger.error("Regime detection failed: %s", e, exc_info=True)
        return {
            "current_regime": "UNKNOWN",
            "regime_statistics": {},
            "error": str(e),
        }


@quant_router.get("/pairs")
async def get_pairs():
    """Get pair trading results and cointegration data."""
    filename, report = _get_latest_report()
    pairs = report.get("pair_trading", {})
    
    return {
        "total_scanned": pairs.get("total_scanned", 0),
        "cointegrated": pairs.get("cointegrated", 0),
        "backtested": pairs.get("backtested", 0),
        "validated": pairs.get("validated", 0),
        "pairs": pairs.get("pairs", []),
        "generated_at": report.get("generated_at"),
    }


@quant_router.get("/fundamentals/{symbol}")
async def get_fundamentals(symbol: str):
    """Get multi-year row-level fundamentals for a specific symbol."""
    from app.data.fundamentals import get_full_fundamentals
    
    try:
        rows = get_full_fundamentals(symbol.upper())
        if not rows:
            raise HTTPException(status_code=404, detail=f"No fundamental data found for {symbol}")
            
        # Group by fiscal year for easy frontend rendering
        data_by_year = {}
        for row in rows:
            if row.fiscal_year not in data_by_year:
                data_by_year[row.fiscal_year] = {}
            data_by_year[row.fiscal_year][row.metric] = {
                "value": row.value,
                "unit": row.unit
            }
            
        return {
            "symbol": symbol.upper(),
            "data": data_by_year
        }
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@quant_router.get("/ohlcv/{symbol}")
async def get_ohlcv(symbol: str, days: int = 180, interval: str = "1d"):
    """
    Get OHLCV data for charting, enriched with simple SMA crossover signals 
    for visual testing. Supports intraday intervals via yfinance.
    """
    from app.data.nse import fetch_nse_ohlcv
    from datetime import date, timedelta
    import pandas as pd
    import yfinance as yf
    
    try:
        if interval != "1d":
            # For intraday, yfinance supports up to 60 days
            limit_days = min(days, 59)
            ticker = yf.Ticker(f"{symbol.upper()}.NS")
            df = ticker.history(period=f"{limit_days}d", interval=interval)
            
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No OHLCV data found for {symbol} at {interval}")
                
            df = df.reset_index()
            # Normalize date column
            date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            if date_col in df.columns:
                df = df.rename(columns={date_col: 'date'})
                
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            # Remove timezone for uniformity
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        else:
            end_date = date.today()
            # Fetch a bit more data for moving average calculation
            start_date = end_date - timedelta(days=days + 50)
            
            df = fetch_nse_ohlcv(symbol.upper(), start_date=start_date, end_date=end_date)
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No OHLCV data found for {symbol}")
                
        # Calculate SMAs for simple signals
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma21'] = df['close'].rolling(window=21).mean()
        
        # Generate signals (1 for buy, -1 for sell, 0 for none)
        df['signal'] = 0
        
        # Buy: SMA10 crosses above SMA21
        buy_condition = (df['sma10'] > df['sma21']) & (df['sma10'].shift(1) <= df['sma21'].shift(1))
        df.loc[buy_condition, 'signal'] = 1
        
        # Sell: SMA10 crosses below SMA21
        sell_condition = (df['sma10'] < df['sma21']) & (df['sma10'].shift(1) >= df['sma21'].shift(1))
        df.loc[sell_condition, 'signal'] = -1
        
        # Filter back down to requested days if it's daily data
        if interval == "1d":
            target_start = pd.Timestamp(date.today() - timedelta(days=days))
            df = df[df['date'] >= target_start]
        
        # Format for lightweight-charts
        formatted_data = []
        for _, row in df.iterrows():
            formatted_data.append({
                "time": int(row['date'].timestamp()),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume']),
                "signal": int(row['signal'])
            })
            
        return {
            "symbol": symbol.upper(),
            "data": formatted_data
        }
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@quant_router.get("/options/{symbol}")
async def get_options_chain(symbol: str):
    """
    Get real NSE options chain (Calls and Puts) with OI, volume, IV.
    
    Sources (in priority order):
      1. nselib — purpose-built NSE scraper
      2. Direct NSE India API with proper headers
      3. Fallback mock if both fail (for UI development only)
    """
    from app.data.nse_fno import get_option_chain
    
    try:
        result = get_option_chain(symbol.upper())
        
        # If real data failed, generate realistic mock as last resort
        if result.get("source") == "none" or not result.get("calls"):
            logger.warning("Real NSE data unavailable for %s, generating mock fallback", symbol)
            return await _generate_mock_options(symbol)
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching options chain for {symbol}: {e}")
        return await _generate_mock_options(symbol)


async def _generate_mock_options(symbol: str):
    """Fallback mock options chain when real NSE data is unavailable."""
    import yfinance as yf
    import numpy as np
    from datetime import date as _date, timedelta
    
    try:
        ticker = yf.Ticker(f"{symbol.upper()}.NS")
        hist = ticker.history(period="1d")
        spot_price = float(hist['Close'].iloc[-1]) if not hist.empty else 2500.0
    except Exception:
        spot_price = 2500.0
    
    if spot_price < 500:
        step = 5
    elif spot_price < 2000:
        step = 10
    elif spot_price < 5000:
        step = 50
    else:
        step = 100
    
    center_strike = round(spot_price / step) * step
    strikes = [center_strike + (i * step) for i in range(-6, 7)]
    calls, puts = [], []
    
    for k in strikes:
        time_value_call = np.random.uniform(2, 20)
        time_value_put = np.random.uniform(2, 20)
        call_price = max(0, spot_price - k) + time_value_call
        put_price = max(0, k - spot_price) + time_value_put
        
        calls.append({
            "strike": k,
            "lastPrice": round(call_price, 2),
            "volume": int(np.random.randint(100, 50000)),
            "openInterest": int(np.random.randint(1000, 200000)),
            "impliedVolatility": round(np.random.uniform(0.12, 0.45), 4),
            "change": round(np.random.uniform(-5, 5), 2),
        })
        puts.append({
            "strike": k,
            "lastPrice": round(put_price, 2),
            "volume": int(np.random.randint(100, 50000)),
            "openInterest": int(np.random.randint(1000, 200000)),
            "impliedVolatility": round(np.random.uniform(0.12, 0.45), 4),
            "change": round(np.random.uniform(-5, 5), 2),
        })
    
    today = _date.today()
    days_ahead = 3 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    expiry = today + timedelta(days=days_ahead)
    
    return {
        "symbol": symbol.upper(),
        "spotPrice": round(spot_price, 2),
        "expiry": expiry.strftime("%Y-%m-%d"),
        "calls": calls,
        "puts": puts,
        "source": "mock_fallback",
        "warning": "Real NSE data unavailable. This is mock data for UI development.",
    }


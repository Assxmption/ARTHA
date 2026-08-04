import pandas as pd
import numpy as np

def generate_mock_options(spot_price, symbol):
    strikes = [round(spot_price * (1 + (i * 0.01))) for i in range(-5, 6)]
    calls = []
    puts = []
    
    for k in strikes:
        call_itm = k < spot_price
        put_itm = k > spot_price
        
        # Simple intrinsic + time value mock
        call_price = max(0, spot_price - k) + np.random.uniform(5, 20)
        put_price = max(0, k - spot_price) + np.random.uniform(5, 20)
        
        calls.append({
            "strike": k,
            "lastPrice": round(call_price, 2),
            "volume": np.random.randint(100, 5000),
            "openInterest": np.random.randint(1000, 20000),
            "impliedVolatility": round(np.random.uniform(0.15, 0.35), 4)
        })
        
        puts.append({
            "strike": k,
            "lastPrice": round(put_price, 2),
            "volume": np.random.randint(100, 5000),
            "openInterest": np.random.randint(1000, 20000),
            "impliedVolatility": round(np.random.uniform(0.15, 0.35), 4)
        })
        
    return {"calls": calls, "puts": puts, "spot": spot_price, "expiry": "2026-07-30"}
print(generate_mock_options(150, "AAPL"))

"""Download/refresh 5-minute Kite history for the ARTHA universe (read-only Kite)."""
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from app.data import intraday_history as ih  # noqa: E402
from app.data.nse_calendar import last_close_boundary  # noqa: E402
from app.data.security_master import PointInTimeSecurityMaster  # noqa: E402

years = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
end = last_close_boundary().date()
start = end - timedelta(days=int(years * 365.25))
master = PointInTimeSecurityMaster.from_file(Path("data_cache/nse/security_master.json"))
symbols = ["NIFTY"] + sorted(master.symbols_at(end))
print(f"Fetching 5-minute bars {start} → {end} for {len(symbols)} symbols", flush=True)
result = ih.update_many(symbols, start, end)
bad = {k: v for k, v in result.items() if not isinstance(v, int)}
print(json.dumps({"ok": sum(isinstance(v, int) for v in result.values()), "failed": bad}, indent=1))

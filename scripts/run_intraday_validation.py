"""Walk-forward OOS validation of the predeclared intraday strategies (writes docs/backtest_reports/intraday_oos_*.json)."""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from app.data.security_master import PointInTimeSecurityMaster  # noqa: E402
from app.data.nse_calendar import last_close_boundary  # noqa: E402
from app.quant import intraday_research as ir  # noqa: E402

t0 = time.time()
master = PointInTimeSecurityMaster.from_file(Path("data_cache/nse/security_master.json"))
symbols = sorted(master.symbols_at(last_close_boundary().date()))
report, path = ir.run(symbols)
print(f"wrote {path} in {time.time() - t0:.0f}s")
print(json.dumps({"data": report["data"], "pairs": report["pairs_selected_on_train"]}, indent=1))
for name, r in report["strategies"].items():
    o, s = r["oos"], r["oos_stress_5bp"]
    print(f"{name:20s} validated={r['validated']!s:5s} OOS Sharpe {o['sharpe_ratio']:+.2f}  ann {o['annualized_return']:+.2%}  "
          f"maxDD {o['max_drawdown']:.2%}  PF {o['profit_factor']:.2f}  | stress Sharpe {s['sharpe_ratio']:+.2f} | "
          f"train Sharpe {r['train']['sharpe_ratio']:+.2f} | orders {r['trades_oos']['orders']}")

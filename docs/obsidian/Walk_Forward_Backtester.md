# Walk-Forward Backtester

Located in `app/quant/backtest.py`.

The testing framework that validates hypotheses from the [[Alpha_Loop]].

- **Vectorized Execution**: Uses vectorized methods (`vectorbt` or pandas) for fast performance over historical data.
- **Realism**: Models transaction costs, slippage, and MCX margin/lot-size mechanics realistic to Indian retail brokerage charges, ensuring backtests aren't frictionless toys.
- **Out-Of-Sample Validation**: Provides the final "Gate." A `QuantSignal` is only eligible for narration when `validated=True`, and that flag is only set after clearing the walk-forward out-of-sample tests here.

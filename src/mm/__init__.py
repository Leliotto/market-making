"""Market making under inventory risk.

Modules
-------
recorder     Record live order book snapshots and trades from Binance.
data         Load recorded data into numpy arrays (Market).
synthetic    Synthetic markets with known parameters, used by the tests.
calibration  Estimate volatility and fill intensity (sigma, A, k).
strategies   Symmetric baseline and Avellaneda-Stoikov (mid- or microprice-centred).
backtest     Event-driven backtester with latency and two fill models.
metrics      PnL, Newey-West t-statistics, paired tests, inventory and markouts.
plots        Standard figures.
"""

__version__ = "0.1.0"

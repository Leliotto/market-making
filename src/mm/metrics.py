"""Performance statistics for a backtest."""

from __future__ import annotations

import numpy as np

from .backtest import BacktestResult
from .calibration import mid_at
from .data import Market

def hac_t_stat(x: np.ndarray, lags: int | None = None) -> float:
    """t-statistic of the mean of x with Newey-West (HAC) standard errors.

    Consecutive PnL changes are not independent (inventory carries over), so the plain
    t-statistic overstates significance. Newey-West corrects for autocorrelation up to `lags`.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    d = x - x.mean()
    var = np.dot(d, d) / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * np.dot(d[lag:], d[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def _sample_every(res: BacktestResult, step_ms: int) -> np.ndarray:
    grid = np.arange(res.t[0], res.t[-1] + 1, step_ms)
    idx = np.clip(np.searchsorted(res.t, grid, side="right") - 1, 0, len(res.t) - 1)
    return res.pnl[idx]


def paired_comparison(a: BacktestResult, b: BacktestResult, step_ms: int = 600_000) -> dict:
    """Is strategy a better than b on the same data? Newey-West t of the difference in PnL changes.

    Both strategies see exactly the same market, so most of the noise cancels in the difference:
    this is a much sharper test than comparing two separate t-statistics.
    """
    da, db = np.diff(_sample_every(a, step_ms)), np.diff(_sample_every(b, step_ms))
    n = min(len(da), len(db))
    return {"pnl_a": float(a.pnl[-1]), "pnl_b": float(b.pnl[-1]),
            "diff": float(a.pnl[-1] - b.pnl[-1]), "t_diff": hac_t_stat(da[:n] - db[:n])}


def summarize(res: BacktestResult, market: Market, markout_s=(1, 10, 60)) -> dict:
    """Key numbers. Markout h = average of side * (mid(t+h) - fill price), per unit.

    A markout that shrinks (or turns negative) as h grows is adverse selection:
    we get filled right before the price moves against us.
    """
    pnl = res.pnl
    lot = res.params["lot_size"]
    f = res.fills
    notional = float(np.sum(f["price"] * f["qty"])) if len(f) else 0.0

    # Newey-West t-statistic of the mean 10-minute PnL change: |t| > 2 means the PnL drift is unlikely
    # to be luck *within this sample*. (An annualised Sharpe on a few hours of data is not meaningful.)
    t_stat = hac_t_stat(np.diff(_sample_every(res, 600_000)))
    drawdown = float(np.max(np.maximum.accumulate(pnl) - pnl))

    out = {
        "strategy": res.strategy,
        "pnl": float(pnl[-1]),
        "fills": int(len(f)),
        "volume_base": float(f["qty"].sum()) if len(f) else 0.0,
        "pnl_bps_of_volume": float(pnl[-1] / notional * 1e4) if notional else float("nan"),
        "pnl_t_stat": t_stat,
        "max_drawdown": drawdown,
        "mean_abs_inv_lots": float(np.mean(np.abs(res.inventory)) / lot),
        "max_abs_inv_lots": float(np.max(np.abs(res.inventory)) / lot),
    }
    if len(f):
        out["spread_capture"] = float(np.mean(f["side"] * (f["mid"] - f["price"])))
        for h in markout_s:
            future_mid = mid_at(market, f["t"] + int(h * 1000))
            out[f"markout_{h}s"] = float(np.mean(f["side"] * (future_mid - f["price"])))
    return out


def format_table(rows: list[dict]) -> str:
    """Plain-text table of several summaries, one row per strategy."""
    if not rows:
        return ""
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:,.4g}"
        return str(v)

    cells = [[fmt(r.get(k, "")) for k in keys] for r in rows]
    widths = [max(len(k), *(len(c[i]) for c in cells)) for i, k in enumerate(keys)]
    line = "  ".join(k.rjust(w) for k, w in zip(keys, widths))
    body = ["  ".join(c[i].rjust(widths[i]) for i in range(len(keys))) for c in cells]
    return "\n".join([line, "-" * len(line), *body])

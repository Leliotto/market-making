"""Standard figures for a backtest."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .backtest import BacktestResult
from .calibration import IntensityFit


def _hours(t_ms: np.ndarray) -> np.ndarray:
    return (t_ms - t_ms[0]) / 3_600_000


def plot_comparison(results: list[BacktestResult], path: str | Path, title: str = "") -> None:
    """PnL and inventory over time for several strategies on the same data."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for r in results:
        h = _hours(r.t)
        ax1.plot(h, r.pnl, lw=1, label=r.strategy)
        ax2.plot(h, r.inventory / r.params["lot_size"], lw=0.8, label=r.strategy)
    ax1.set_ylabel("PnL (mark to mid)")
    ax2.set_ylabel("inventory (lots)")
    ax2.set_xlabel("hours")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_quotes(res: BacktestResult, path: str | Path, minutes: float = 10) -> None:
    """Zoom on the first minutes: mid, our bid/ask and fills."""
    end = res.t[0] + minutes * 60_000
    w = res.t <= end
    h = (res.t[w] - res.t[0]) / 60_000
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(h, res.mid[w], color="0.3", lw=1, label="mid")
    ax.plot(h, res.bid[w], color="tab:green", lw=0.8, label="our bid")
    ax.plot(h, res.ask[w], color="tab:red", lw=0.8, label="our ask")
    f = res.fills[res.fills["t"] <= end]
    fh = (f["t"] - res.t[0]) / 60_000
    ax.scatter(fh[f["side"] > 0], f["price"][f["side"] > 0], marker="^", color="tab:green", s=25, label="bought")
    ax.scatter(fh[f["side"] < 0], f["price"][f["side"] < 0], marker="v", color="tab:red", s=25, label="sold")
    ax.set_xlabel("minutes")
    ax.set_title(res.strategy)
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_intensity(fit: IntensityFit, path: str | Path, title: str = "") -> None:
    """Empirical fill intensity vs the exponential fit (log scale: a straight line if the model holds)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ok = fit.rates > 0
    ax.semilogy(fit.deltas[ok & fit.used], fit.rates[ok & fit.used], "o", ms=4, label="empirical (used in fit)")
    if np.any(ok & ~fit.used):
        ax.semilogy(fit.deltas[ok & ~fit.used], fit.rates[ok & ~fit.used], "o", ms=4, mfc="none", color="0.5", label="too few fills")
    ax.semilogy(fit.deltas, fit.A * np.exp(-fit.k * fit.deltas), "-", label=f"A e^(-k d), A={fit.A:.3g}, k={fit.k:.3g}")
    ax.set_xlabel("distance from mid")
    ax.set_ylabel("fills per second per side")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

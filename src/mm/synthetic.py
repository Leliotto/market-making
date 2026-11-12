"""Synthetic market with known parameters, used to test calibration and the backtester.

Mid price: arithmetic Brownian motion with volatility sigma (price units per sqrt(second)).
Trades: on each side, market orders arrive as a Poisson process with rate A per second, and
each reaches a distance delta ~ Exponential(k) from the mid. So the rate of trades that reach
at least delta is A * exp(-k * delta), exactly the Avellaneda-Stoikov intensity model.
"""

from __future__ import annotations

import numpy as np

from .data import Market


def make_market(
    hours: float = 1.0,
    s0: float = 100.0,
    sigma: float = 0.02,
    A: float = 2.0,
    k: float = 20.0,
    tick: float = 0.01,
    half_spread: float = 0.005,
    dt_ms: int = 100,
    trade_qty: float = 1.0,
    seed: int = 0,
) -> Market:
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * 1000 / dt_ms)
    book_t = np.arange(n, dtype=np.int64) * dt_ms
    dt = dt_ms / 1000.0
    mid = s0 + np.concatenate([[0.0], np.cumsum(rng.normal(0.0, sigma * np.sqrt(dt), n - 1))])
    # quote the touch on the tick grid around the true mid
    bid = np.floor((mid - half_spread) / tick) * tick
    ask = bid + max(tick, np.round(2 * half_spread / tick) * tick)

    t_list, px_list, side_list = [], [], []
    T = n * dt
    for side in (+1, -1):
        m = rng.poisson(A * T)
        t = np.sort(rng.uniform(0, T, m))
        idx = np.minimum((t / dt).astype(int), n - 1)
        delta = rng.exponential(1.0 / k, m)
        t_list.append((t * 1000).astype(np.int64))
        px_list.append(mid[idx] + side * delta)
        side_list.append(np.full(m, side, dtype=np.int8))
    tt = np.concatenate(t_list)
    order = np.argsort(tt, kind="stable")
    px = np.concatenate(px_list)[order]
    sd = np.concatenate(side_list)[order]
    return Market(
        book_t=book_t,
        bid_px=bid[:, None], bid_qty=np.ones((n, 1)),
        ask_px=ask[:, None], ask_qty=np.ones((n, 1)),
        trade_t=tt[order], trade_px=px, trade_qty=np.full(len(px), trade_qty), trade_side=sd,
    )

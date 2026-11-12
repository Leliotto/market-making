"""Estimate the Avellaneda-Stoikov inputs from data.

sigma : volatility of the mid price, in price units per sqrt(second)
A, k  : fill intensity lambda(delta) = A * exp(-k * delta): the rate (per second, per side)
        at which a quote resting `delta` away from the mid gets filled.

Two estimators for (A, k):

estimate_intensity          (recommended) Measures fills the way the backtester does.
    For every quote interval (one book snapshot to the next, shifted by the latency) it takes
    the mid at the start of the interval and asks: did a seller trade more than delta below it
    (or a buyer more than delta above it)? The fraction of intervals with such a trade, turned
    into a Poisson rate by maximum likelihood, is lambda(delta). The fit is then consistent with
    the fill model used in the backtest.

estimate_intensity_trades   (naive, kept for comparison) Distance of each trade from the
    mid of the last snapshot. On fast markets this mostly measures how far the price moved
    between the snapshot and the trade, so k comes out far too small (spreads far too wide).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Market


def mid_at(market: Market, t_ms: np.ndarray) -> np.ndarray:
    """Mid price of the last book snapshot at or before each time."""
    idx = np.searchsorted(market.book_t, t_ms, side="right") - 1
    idx = np.clip(idx, 0, len(market.book_t) - 1)
    return market.mid[idx]


def estimate_sigma(market: Market, dt_s: float = 1.0) -> float:
    """Std of mid-price changes sampled every dt_s seconds, scaled to one second."""
    grid = np.arange(market.book_t[0], market.book_t[-1], int(dt_s * 1000))
    m = mid_at(market, grid)
    return float(np.std(np.diff(m)) / np.sqrt(dt_s))


@dataclass
class IntensityFit:
    A: float
    k: float
    deltas: np.ndarray   # distance grid used for the fit
    rates: np.ndarray    # empirical lambda(delta), fills per second per side
    used: np.ndarray     # boolean mask of the points used in the regression
    method: str = "window"


def _fit(deltas: np.ndarray, rates: np.ndarray, used: np.ndarray, method: str) -> IntensityFit:
    if used.sum() < 3:
        raise ValueError("too few usable points to fit k; record more data or use a less liquid pair")
    slope, intercept = np.polyfit(deltas[used], np.log(rates[used]), 1)
    return IntensityFit(A=float(np.exp(intercept)), k=float(-slope), deltas=deltas, rates=rates, used=used, method=method)


def _poisson_rate(hit_dt: np.ndarray, total_dt: float) -> float:
    """MLE of a Poisson rate from intervals of different lengths.

    Each interval i lasts dt_i; it is a 'hit' if at least one event happened, which has
    probability 1 - exp(-lambda * dt_i). Solves d(loglik)/d(lambda) = 0 by bisection.
    """
    if len(hit_dt) == 0:
        return 0.0
    miss_dt = total_dt - hit_dt.sum()
    if miss_dt <= 0:
        return float("inf")

    def score(lam):
        x = lam * hit_dt
        return float(np.sum(hit_dt / np.expm1(x))) - miss_dt  # decreasing in lam

    lo, hi = 1e-9, 1e4
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def interval_reach(market: Market, latency_ms: int = 50):
    """For each quote interval: how far below / above the starting mid sellers / buyers traded.

    Returns (dt_seconds, sell_reach, buy_reach); reach is -inf when no trade of that side happened.
    """
    bt = market.book_t
    start = bt + latency_ms
    dt = np.diff(bt).astype(float) / 1000.0           # interval i lasts until the next snapshot
    n = len(dt)
    idx = np.searchsorted(start, market.trade_t, side="right") - 1
    ok = (idx >= 0) & (idx < n)
    idx, px, side = idx[ok], market.trade_px[ok], market.trade_side[ok]
    mid0 = market.mid[idx]
    sell_reach = np.full(n, -np.inf)
    buy_reach = np.full(n, -np.inf)
    s, b = side < 0, side > 0
    np.maximum.at(sell_reach, idx[s], mid0[s] - px[s])
    np.maximum.at(buy_reach, idx[b], px[b] - mid0[b])
    return dt, sell_reach, buy_reach


def estimate_intensity(
    market: Market,
    latency_ms: int = 50,
    fill_mode: str = "through",
    deltas: np.ndarray | None = None,
    n_points: int = 30,
    min_hits: int = 30,
) -> IntensityFit:
    """Fill intensity per side, measured on quote intervals (see module docstring)."""
    dt, sell_reach, buy_reach = interval_reach(market, latency_ms)
    total = float(dt.sum())
    if deltas is None:
        reach = np.concatenate([sell_reach, buy_reach])
        pos = reach[np.isfinite(reach) & (reach > 0)]
        if len(pos) == 0:
            raise ValueError("no trades away from the mid: cannot fit intensity")
        deltas = np.linspace(0.0, np.quantile(pos, 0.995), n_points)
    rates, hits = [], []
    eps = 1e-12
    for d in deltas:
        r = []
        h = 0
        for reach in (sell_reach, buy_reach):
            hit = reach > d + eps if fill_mode == "through" else reach >= d - eps
            r.append(_poisson_rate(dt[hit], total))
            h += int(hit.sum())
        rates.append(0.5 * (r[0] + r[1]))
        hits.append(h)
    rates, hits = np.array(rates), np.array(hits)
    used = (hits >= min_hits) & np.isfinite(rates) & (rates > 0)
    return _fit(deltas, rates, used, "window")


def estimate_intensity_trades(
    market: Market, deltas: np.ndarray | None = None, n_points: int = 30, min_count: int = 20
) -> IntensityFit:
    """Naive estimator: distance of each trade from the mid of the last snapshot."""
    mid = mid_at(market, market.trade_t)
    dist = market.trade_side * (market.trade_px - mid)
    T = market.duration_s
    if deltas is None:
        pos = dist[dist > 0]
        if len(pos) == 0:
            raise ValueError("no trades away from the mid: cannot fit intensity")
        deltas = np.linspace(0.0, np.quantile(pos, 0.99), n_points)
    counts = np.array([(dist >= d).sum() for d in deltas])
    rates = counts / T / 2.0
    return _fit(deltas, rates, counts >= min_count, "trades")

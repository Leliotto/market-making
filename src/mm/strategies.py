"""Quoting strategies.

A strategy looks at the current state and returns a bid and an ask price
(None means: do not quote that side). Inventory q is measured in lots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class State:
    t_ms: int
    mid: float
    best_bid: float
    best_ask: float
    q: float        # inventory in lots (positive = long)
    tick: float
    bid_qty: float = 1.0   # size at the best bid
    ask_qty: float = 1.0   # size at the best ask

    @property
    def microprice(self) -> float:
        """Mid weighted by the opposite side's size: leans towards the thinner side of the book,
        which is where the price tends to move next (Stoikov, 'The micro-price', 2018)."""
        tot = self.bid_qty + self.ask_qty
        if tot <= 0:
            return self.mid
        return (self.best_bid * self.ask_qty + self.best_ask * self.bid_qty) / tot


def _round_quotes(bid: float | None, ask: float | None, s: State) -> tuple[float | None, float | None]:
    """Snap to the tick grid (bid down, ask up) and never cross the book: we only post passive orders."""
    if bid is not None:
        bid = math.floor(bid / s.tick + 1e-9) * s.tick
        bid = min(bid, s.best_ask - s.tick)
    if ask is not None:
        ask = math.ceil(ask / s.tick - 1e-9) * s.tick
        ask = max(ask, s.best_bid + s.tick)
    if bid is not None and ask is not None and ask < bid + s.tick - 1e-12:
        ask = bid + s.tick  # our own bid and ask must not touch
    return bid, ask


class Strategy:
    name = "strategy"

    def quote(self, s: State) -> tuple[float | None, float | None]:
        raise NotImplementedError


class Symmetric(Strategy):
    """Baseline: fixed half-spread around the mid, ignores inventory (except a hard limit)."""

    def __init__(self, half_spread: float, q_max: float = 10):
        self.half_spread = half_spread
        self.q_max = q_max
        self.name = f"symmetric(h={half_spread:g})"

    def quote(self, s: State):
        bid = s.mid - self.half_spread if s.q < self.q_max - 1e-6 else None
        ask = s.mid + self.half_spread if s.q > -self.q_max + 1e-6 else None
        return _round_quotes(bid, ask, s)


class AvellanedaStoikov(Strategy):
    """Avellaneda & Stoikov (2008), 'High-frequency trading in a limit order book'.

    reservation price  r = s - q * gamma * sigma^2 * tau
    total spread       d = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)
    quotes             bid = r - d/2,  ask = r + d/2

    Crypto trades 24/7, so instead of a fixed end time T we use a rolling horizon:
    tau is a constant number of seconds (how far ahead the market maker cares about risk).

    Units: prices in quote currency, sigma in price/sqrt(s), k and gamma in 1/price, q in lots.
    With center="micro" the quotes are built around the microprice instead of the mid.
    """

    def __init__(self, gamma: float, sigma: float, k: float, horizon_s: float = 60.0, q_max: float = 10,
                 center: str = "mid"):
        if center not in ("mid", "micro"):
            raise ValueError("center must be 'mid' or 'micro'")
        self.gamma, self.sigma, self.k = gamma, sigma, k
        self.tau = horizon_s
        self.q_max = q_max
        self.center = center
        self.name = f"AS{'-micro' if center == 'micro' else ''}(gamma={gamma:.3g})"

    @classmethod
    def from_skew(cls, skew_ticks: float, tick: float, sigma: float, k: float, horizon_s: float = 60.0,
                  q_max: float = 10, center: str = "mid") -> "AvellanedaStoikov":
        """Choose gamma so that each lot of inventory moves the reservation price by `skew_ticks` ticks.

        gamma * sigma^2 * tau is the shift per lot, so gamma = skew_ticks * tick / (sigma^2 * tau).
        Unlike a raw gamma, "ticks per lot" means the same thing on every coin.
        """
        gamma = skew_ticks * tick / (sigma**2 * horizon_s)
        s = cls(gamma, sigma, k, horizon_s, q_max, center)
        s.skew_ticks = skew_ticks
        s.name = f"AS{'-micro' if center == 'micro' else ''}(skew={skew_ticks:g}t)"
        return s

    def reservation_price(self, mid: float, q: float) -> float:
        return mid - q * self.gamma * self.sigma**2 * self.tau

    def total_spread(self) -> float:
        g = self.gamma
        return g * self.sigma**2 * self.tau + (2.0 / g) * math.log1p(g / self.k)

    def quote(self, s: State):
        r = self.reservation_price(s.microprice if self.center == "micro" else s.mid, s.q)
        half = 0.5 * self.total_spread()
        bid = r - half if s.q < self.q_max - 1e-6 else None
        ask = r + half if s.q > -self.q_max + 1e-6 else None
        return _round_quotes(bid, ask, s)

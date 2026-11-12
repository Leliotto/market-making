"""Event-driven backtester for a single passive market maker.

At every book snapshot the strategy posts one bid and one ask of `lot_size`.
New quotes reach the exchange `latency_ms` later; until then the previous quotes stay live.
Between two quote updates, each recorded trade is matched against our live quotes.

Fill model (we never see our own place in the queue, so we have to assume something):
  "through"  a resting bid fills only if a seller trades at a price strictly BELOW it
             (everyone at our price must have been filled first). Conservative.
  "touch"    a bid fills if a seller trades AT or below it. Optimistic: it assumes we are
             always first in the queue.
Fills happen at our own quoted price. Fees are charged on notional (negative = rebate).
Our orders are assumed too small to move the market (no market impact).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Market
from .strategies import State, Strategy


@dataclass
class BacktestResult:
    strategy: str
    t: np.ndarray            # snapshot times (ms)
    mid: np.ndarray
    bid: np.ndarray          # our quotes (nan = not quoting)
    ask: np.ndarray
    inventory: np.ndarray    # base units
    cash: np.ndarray
    fills: np.ndarray        # structured: t, side (+1 bought / -1 sold), price, qty, mid
    params: dict = field(default_factory=dict)

    @property
    def pnl(self) -> np.ndarray:
        """Mark-to-mid PnL in quote currency."""
        return self.cash + self.inventory * self.mid


FILL_DTYPE = np.dtype([("t", "i8"), ("side", "i1"), ("price", "f8"), ("qty", "f8"), ("mid", "f8")])


def run_backtest(
    market: Market,
    strategy: Strategy,
    lot_size: float,
    tick: float,
    maker_fee_bps: float = 0.0,
    fill_mode: str = "through",
    latency_ms: int = 50,
) -> BacktestResult:
    if fill_mode not in ("through", "touch"):
        raise ValueError("fill_mode must be 'through' or 'touch'")
    through = fill_mode == "through"
    eps = tick * 1e-6
    fee = maker_fee_bps / 1e4

    bt, mids = market.book_t, market.mid
    bb, ba = market.best_bid, market.best_ask
    bq, aq = market.bid_qty[:, 0], market.ask_qty[:, 0]
    tt, tp, tq, ts = market.trade_t, market.trade_px, market.trade_qty, market.trade_side
    n, m = len(bt), len(tt)

    out_bid = np.full(n, np.nan)
    out_ask = np.full(n, np.nan)
    out_inv = np.zeros(n)
    out_cash = np.zeros(n)
    fills = []

    inv = cash = 0.0
    bid = ask = None          # live quotes
    rem_bid = rem_ask = 0.0   # unfilled size of the live quotes
    last_mid = mids[0]
    j = 0
    for i in range(n):
        # 1) trades that happen before this snapshot's quotes go live hit the previous quotes
        active_from = bt[i] + latency_ms
        while j < m and tt[j] < active_from:
            side, px = ts[j], tp[j]
            mid_now = mids[i] if tt[j] >= bt[i] else last_mid
            if side < 0 and bid is not None and rem_bid > 0:
                hit = px < bid - eps if through else px <= bid + eps
                if hit:
                    q = rem_bid if through else min(rem_bid, tq[j])
                    inv += q
                    cash -= bid * q * (1 + fee)
                    rem_bid -= q
                    fills.append((tt[j], 1, bid, q, mid_now))
            elif side > 0 and ask is not None and rem_ask > 0:
                hit = px > ask + eps if through else px >= ask - eps
                if hit:
                    q = rem_ask if through else min(rem_ask, tq[j])
                    inv -= q
                    cash += ask * q * (1 - fee)
                    rem_ask -= q
                    fills.append((tt[j], -1, ask, q, mid_now))
            j += 1

        # 2) record the state at this snapshot
        last_mid = mids[i]
        out_inv[i], out_cash[i] = inv, cash

        # 3) new quotes (cancel-and-replace every snapshot, full size again)
        bid, ask = strategy.quote(State(int(bt[i]), mids[i], bb[i], ba[i], inv / lot_size, tick, bq[i], aq[i]))
        rem_bid = lot_size if bid is not None else 0.0
        rem_ask = lot_size if ask is not None else 0.0
        out_bid[i] = np.nan if bid is None else bid
        out_ask[i] = np.nan if ask is None else ask

    return BacktestResult(
        strategy=strategy.name, t=bt, mid=mids, bid=out_bid, ask=out_ask,
        inventory=out_inv, cash=out_cash, fills=np.array(fills, dtype=FILL_DTYPE),
        params=dict(lot_size=lot_size, tick=tick, maker_fee_bps=maker_fee_bps,
                    fill_mode=fill_mode, latency_ms=latency_ms),
    )

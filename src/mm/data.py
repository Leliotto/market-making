"""Load recorded data into a single `Market` object of numpy arrays."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Market:
    """Order book snapshots and trades for one symbol, both sorted by time.

    Times are milliseconds (int64), all measured on the same local clock.
    Book arrays have shape (n_snapshots, levels); level 0 is the best price.
    trade_side is the aggressor side: +1 buyer lifted the ask, -1 seller hit the bid.
    """

    book_t: np.ndarray
    bid_px: np.ndarray
    bid_qty: np.ndarray
    ask_px: np.ndarray
    ask_qty: np.ndarray
    trade_t: np.ndarray
    trade_px: np.ndarray
    trade_qty: np.ndarray
    trade_side: np.ndarray

    @property
    def best_bid(self) -> np.ndarray:
        return self.bid_px[:, 0]

    @property
    def best_ask(self) -> np.ndarray:
        return self.ask_px[:, 0]

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def duration_s(self) -> float:
        return float(self.book_t[-1] - self.book_t[0]) / 1000.0

    def slice(self, start_ms: int | None = None, end_ms: int | None = None) -> "Market":
        """Sub-market between two times (ms), e.g. to split train and test periods."""
        lo = -np.inf if start_ms is None else start_ms
        hi = np.inf if end_ms is None else end_ms
        b = (self.book_t >= lo) & (self.book_t < hi)
        tr = (self.trade_t >= lo) & (self.trade_t < hi)
        return Market(self.book_t[b], self.bid_px[b], self.bid_qty[b], self.ask_px[b], self.ask_qty[b],
                      self.trade_t[tr], self.trade_px[tr], self.trade_qty[tr], self.trade_side[tr])

    def summary(self) -> str:
        spread = self.best_ask - self.best_bid
        return (f"{len(self.book_t):,} book snapshots, {len(self.trade_t):,} trades over "
                f"{self.duration_s / 3600:.2f} h | mid {self.mid[0]:.2f} -> {self.mid[-1]:.2f} | "
                f"median spread {np.median(spread):.4f}")


def _iter_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        try:
            for line in fh:
                yield line
        except EOFError:
            # a gzip file cut short (e.g. the machine went to sleep): keep what we have
            return


def load_market(data_dir: str | Path, symbol: str, hours: list[str] | None = None) -> Market:
    """Load all files in data_dir/<symbol>/ (or only the given 'YYYYmmdd_HH' hours)."""
    folder = Path(data_dir) / symbol.lower()
    files = sorted(list(folder.glob("*.jsonl.gz")) + list(folder.glob("*.jsonl")))
    if hours is not None:
        files = [f for f in files if f.name.split(".")[0][:11] in set(hours)]
    if not files:
        raise FileNotFoundError(f"no recorded files in {folder}")

    # parse one hourly file at a time and convert it to numpy straight away:
    # a day of nested Python lists would need several GB of RAM, numpy arrays a few hundred MB
    parts_t, parts_bid, parts_ask, parts_tr = [], [], [], []
    for f in files:
        book_t, bids, asks, trades = [], [], [], []
        for line in _iter_lines(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line
            if rec["type"] == "book":
                if rec["b"] and rec["a"]:
                    book_t.append(rec["t"])
                    bids.append(rec["b"])
                    asks.append(rec["a"])
            elif rec["type"] == "trade":
                trades.append((rec["t"], rec["p"], rec["q"], rec["s"]))
        if book_t:
            lv = min(min(len(b) for b in bids), min(len(a) for a in asks))
            parts_t.append(np.asarray(book_t, dtype=np.int64))
            parts_bid.append(np.array([b[:lv] for b in bids], dtype=float))
            parts_ask.append(np.array([a[:lv] for a in asks], dtype=float))
        parts_tr.append(np.array(trades, dtype=float).reshape(-1, 4))
    if not parts_t:
        raise ValueError(f"no order book snapshots in {folder}")

    levels = min(min(b.shape[1] for b in parts_bid), min(a.shape[1] for a in parts_ask))
    book_t = np.concatenate(parts_t)
    bid = np.concatenate([b[:, :levels] for b in parts_bid])
    ask = np.concatenate([a[:, :levels] for a in parts_ask])
    tr = np.concatenate(parts_tr)

    ob = np.argsort(np.asarray(book_t), kind="stable")
    ot = np.argsort(tr[:, 0], kind="stable")
    return Market(
        book_t=np.asarray(book_t, dtype=np.int64)[ob],
        bid_px=bid[ob, :, 0], bid_qty=bid[ob, :, 1],
        ask_px=ask[ob, :, 0], ask_qty=ask[ob, :, 1],
        trade_t=tr[ot, 0].astype(np.int64), trade_px=tr[ot, 1],
        trade_qty=tr[ot, 2], trade_side=tr[ot, 3].astype(np.int8),
    )

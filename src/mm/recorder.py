"""Record Binance spot order book snapshots and trades to disk.

Uses Binance's free public websocket streams (no API key needed):
  <symbol>@depth<levels>@100ms   top-of-book snapshot every 100 ms
  <symbol>@aggTrade              every aggregated trade

Output: one JSON line per event, one file per symbol per hour:
  data/raw/<symbol>/<YYYYmmdd_HH>.jsonl   (gzipped when the hour is closed)

  {"t": 1731060000123, "type": "book", "b": [[price, qty], ...], "a": [[price, qty], ...]}
  {"t": 1731060000150, "type": "trade", "p": 76012.5, "q": 0.013, "s": 1, "T": 1731060000149}

`t` is the local receive time in ms (depth snapshots carry no exchange time),
`T` the exchange trade time, and `s` the aggressor side: +1 buyer, -1 seller.

Usage:
  python -m mm.recorder --symbols btcusdt ethusdt --hours 24
Stop any time with Ctrl+C: the current file is closed and compressed.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

BASE_URL = "wss://stream.binance.com:9443/stream?streams="


class HourlyWriter:
    """Append JSON lines to one file per hour; gzip each file once it is closed."""

    def __init__(self, root: Path, symbol: str):
        self.dir = root / symbol
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hour: str | None = None
        self.fh = None
        self.lines = 0

    def write(self, t_ms: int, record: dict) -> None:
        hour = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime("%Y%m%d_%H")
        if hour != self.hour:
            self.close()
            self.hour = hour
            self.fh = open(self.dir / f"{hour}.jsonl", "a", encoding="utf-8")
        self.fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.lines += 1

    def close(self) -> None:
        if self.fh is None:
            return
        path = Path(self.fh.name)
        self.fh.close()
        self.fh = None
        gz = path.with_suffix(".jsonl.gz")
        # if an earlier run already compressed this hour, keep both parts
        if gz.exists():
            gz = path.with_name(f"{path.stem}_{int(time.time())}.jsonl.gz")
        with open(path, "rb") as src, gzip.open(gz, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()


def parse_message(msg: dict, levels: int) -> tuple[str, dict] | None:
    """Turn one combined-stream message into (symbol, record). Returns None if unknown."""
    stream: str = msg.get("stream", "")
    data = msg.get("data", {})
    symbol = stream.split("@")[0]
    now = int(time.time() * 1000)
    if "@depth" in stream:
        return symbol, {
            "t": now,
            "type": "book",
            "b": [[float(p), float(q)] for p, q in data["bids"][:levels]],
            "a": [[float(p), float(q)] for p, q in data["asks"][:levels]],
        }
    if stream.endswith("@aggTrade"):
        # m = True means the buyer was the maker, so the aggressor was a seller
        return symbol, {
            "t": now,
            "type": "trade",
            "p": float(data["p"]),
            "q": float(data["q"]),
            "s": -1 if data["m"] else 1,
            "T": int(data["T"]),
        }
    return None


async def record(symbols: list[str], out_dir: Path, levels: int, hours: float | None) -> None:
    depth = 5 if levels <= 5 else 10 if levels <= 10 else 20
    streams = []
    for s in symbols:
        streams += [f"{s}@depth{depth}@100ms", f"{s}@aggTrade"]
    url = BASE_URL + "/".join(streams)
    writers = {s: HourlyWriter(out_dir, s) for s in symbols}
    deadline = None if hours is None else time.time() + hours * 3600
    backoff = 1.0
    last_log = time.time()
    try:
        while deadline is None or time.time() < deadline:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_size=2**22) as ws:
                    print(f"connected: {', '.join(streams)}")
                    backoff = 1.0
                    async for raw in ws:
                        parsed = parse_message(json.loads(raw), levels)
                        if parsed:
                            symbol, rec = parsed
                            writers[symbol].write(rec["t"], rec)
                        if time.time() - last_log > 60:
                            last_log = time.time()
                            counts = ", ".join(f"{s}: {w.lines}" for s, w in writers.items())
                            print(f"{datetime.now():%H:%M:%S} lines written  {counts}")
                        if deadline is not None and time.time() >= deadline:
                            break
            except (websockets.WebSocketException, OSError, asyncio.TimeoutError) as exc:
                # Binance drops every connection after 24h; network blips happen too
                print(f"connection lost ({exc!r}); reconnecting in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
    finally:
        for w in writers.values():
            w.close()
        print("files closed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["btcusdt"], help="lowercase Binance spot symbols")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--levels", type=int, default=10, help="book levels to keep per side (max 20)")
    ap.add_argument("--hours", type=float, default=None, help="stop after this many hours (default: run until Ctrl+C)")
    args = ap.parse_args()
    symbols = [s.lower() for s in args.symbols]
    try:
        asyncio.run(record(symbols, args.out, args.levels, args.hours))
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()

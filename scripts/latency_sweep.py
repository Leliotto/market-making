"""How much does speed matter? Same quotes, different execution latency.

The strategy is calibrated once on the train period (at --calib-latency), so its quotes are
identical in every run. Only the latency changes: the time between deciding a quote (at a book
snapshot) and the quote being live on the exchange. Any difference in results is then caused by
speed alone. (--recalibrate re-fits the strategy at each latency instead, which answers a
different question: how well can a slower trader do if it adapts its quotes?)

Caveat: the recorded book arrives in 100 ms snapshots, so the strategy never reacts faster
than the data allows. Differences between 0 and 10 ms are hard to measure; the sweep shows the
trend, not the true value of microsecond speed.

Example:
  python scripts/latency_sweep.py --symbol solusdt --notional 100 --data-dir C:\\mm_data --fit-ticks 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mm.backtest import run_backtest
from mm.calibration import estimate_intensity, estimate_sigma
from mm.data import load_market
from mm.metrics import format_table, paired_comparison, summarize
from mm.strategies import AvellanedaStoikov
from mm.synthetic import make_market

KEEP = ["latency_ms", "fill", "strategy", "pnl", "fills", "pnl_bps_of_volume", "pnl_t_stat",
        "mean_abs_inv_lots", "spread_capture", "markout_1s", "markout_10s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="btcusdt")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lot", type=float, default=1.0)
    ap.add_argument("--notional", type=float, default=None)
    ap.add_argument("--tick", type=float, default=None)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--skew-ticks", type=float, default=2.0, help="ticks of quote shift per lot of inventory")
    ap.add_argument("--q-max", type=float, default=10)
    ap.add_argument("--fit-ticks", type=int, default=None)
    ap.add_argument("--latencies", type=int, nargs="+", default=[0, 10, 50, 200])
    ap.add_argument("--calib-latency", type=int, default=50, help="latency assumed when calibrating (fixed quotes)")
    ap.add_argument("--recalibrate", action="store_true", help="re-fit the strategy at every latency")
    ap.add_argument("--fill-mode", choices=["both", "through", "touch"], default="both")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    from run_backtest import infer_tick  # same folder

    if args.synthetic:
        market, name = make_market(hours=4.0, seed=1), "synthetic"
    else:
        market, name = load_market(args.data_dir, args.symbol), args.symbol
    tick = args.tick or infer_tick(market)
    lot = float(f"{args.notional / float(np.median(market.mid)):.3g}") if args.notional else args.lot
    print(f"{name}: {market.summary()} | tick {tick:g} | lot {lot:g}")
    print("mode:", "re-calibrated at every latency" if args.recalibrate
          else f"fixed quotes, calibrated once at {args.calib_latency} ms")

    split = int(market.book_t[0] + args.train_frac * (market.book_t[-1] - market.book_t[0]))
    train, test = market.slice(None, split), market.slice(split, None)
    sigma = estimate_sigma(train)
    deltas = (np.arange(args.fit_ticks) + 0.5) * tick if args.fit_ticks else None
    modes = ["through", "touch"] if args.fill_mode == "both" else [args.fill_mode]

    def build(mode: str, lat: int):
        fit = estimate_intensity(train, latency_ms=lat, fill_mode=mode, deltas=deltas)
        return fit, [AvellanedaStoikov.from_skew(args.skew_ticks, tick, sigma, fit.k, args.horizon, args.q_max, center=c)
                     for c in ("mid", "micro")]

    rows, pairs = [], []
    for mode in modes:
        if not args.recalibrate:
            fit, strats = build(mode, args.calib_latency)
            print(f"{mode}: k={fit.k:.4g}, total spread {strats[0].total_spread() / tick:.1f} ticks")
        for lat in args.latencies:
            if args.recalibrate:
                fit, strats = build(mode, lat)
            res = []
            for s in strats:
                r = run_backtest(test, s, lot_size=lot, tick=tick, fill_mode=mode, latency_ms=lat)
                res.append(r)
                full = {"latency_ms": lat, "fill": mode, **summarize(r, test)}
                rows.append({k: full.get(k, "") for k in KEEP})
            c = paired_comparison(res[1], res[0])
            pairs.append({"latency_ms": lat, "fill": mode, "micro_minus_AS": c["diff"], "t_diff": c["t_diff"]})
            print(f"done: latency {lat} ms, {mode}")

    print()
    print(format_table(rows))
    print("\nmicroprice vs mid on the same data (t_diff > 2: micro significantly better, < -2: worse)")
    print(format_table(pairs))
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_latency_sweep{'_recalibrated' if args.recalibrate else ''}"
    with open(args.out / f"{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEEP)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {args.out / (tag + '.csv')}")
    print("caveat: book snapshots arrive every 100 ms, so latencies below that are only partly measurable.")


if __name__ == "__main__":
    main()

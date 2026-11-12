"""Calibrate on the first part of the data, then backtest on the rest.

Splitting train and test in time avoids look-ahead: the strategy parameters are
estimated only from data the market maker would have seen before trading.

By default every strategy runs under both fill models, which bracket reality:
  through  conservative: filled only when the market trades beyond our price
  touch    optimistic:   filled when the market trades at our price (first in queue)

Examples:
  python scripts/run_backtest.py --synthetic
  python scripts/run_backtest.py --symbol solusdt --notional 100 --data-dir C:\\mm_data
  python scripts/run_backtest.py --symbol solusdt --notional 100 --data-dir C:\\mm_data --fit-ticks 10
  python scripts/run_backtest.py --symbol btcusdt --lot 0.001 --fill-mode through
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mm.backtest import run_backtest
from mm.calibration import estimate_intensity, estimate_intensity_trades, estimate_sigma
from mm.data import Market, load_market
from mm.metrics import format_table, paired_comparison, summarize
from mm.plots import plot_comparison, plot_intensity, plot_quotes
from mm.strategies import AvellanedaStoikov, Symmetric
from mm.synthetic import make_market


def infer_tick(market: Market) -> float:
    """Smallest gap between distinct quoted prices (assumes some adjacent levels were observed)."""
    px = np.round(np.concatenate([market.bid_px.ravel(), market.ask_px.ravel()]), 8)
    d = np.diff(np.unique(px))
    return float(np.round(d[d > 1e-9].min(), 8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="btcusdt")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--synthetic", action="store_true", help="use a synthetic market instead of recorded data")
    ap.add_argument("--lot", type=float, default=1.0, help="order size in base units (e.g. 0.001 BTC)")
    ap.add_argument("--notional", type=float, default=None,
                    help="order size in quote currency (e.g. 100 USDT); overrides --lot, uses the median mid")
    ap.add_argument("--tick", type=float, default=None, help="price tick (default: inferred from the data)")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--horizon", type=float, default=60.0, help="AS risk horizon tau in seconds")
    ap.add_argument("--skew-ticks", type=float, nargs="+", default=[0.5, 2.0, 5.0, 20.0],
                    help="risk aversions to test, as ticks of quote shift per lot of inventory")
    ap.add_argument("--q-max", type=float, default=10, help="hard inventory limit in lots")
    ap.add_argument("--fill-mode", choices=["both", "through", "touch"], default="both")
    ap.add_argument("--fit-ticks", type=int, default=None,
                    help="fit the fill intensity only on the first N price levels away from the mid "
                         "(quotes at 0.5, 1.5, ... ticks); default: the whole range of observed distances")
    ap.add_argument("--no-micro", action="store_true", help="skip the microprice-centred AS variants")
    ap.add_argument("--latency", type=int, default=50, help="ms between deciding a quote and it being live")
    ap.add_argument("--fee-bps", type=float, default=0.0, help="maker fee in bps of notional (negative = rebate)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    if args.synthetic:
        market, name = make_market(hours=4.0, seed=1), "synthetic"
    else:
        market, name = load_market(args.data_dir, args.symbol), args.symbol
    tick = args.tick or infer_tick(market)
    lot = args.lot
    if args.notional:
        lot = float(f"{args.notional / float(np.median(market.mid)):.3g}")
    print(f"{name}: {market.summary()} | tick {tick:g} | lot {lot:g}")

    split = int(market.book_t[0] + args.train_frac * (market.book_t[-1] - market.book_t[0]))
    train, test = market.slice(None, split), market.slice(split, None)
    sigma = estimate_sigma(train)
    print(f"sigma on train: {sigma:.4g} per sqrt(s)")
    try:
        naive = estimate_intensity_trades(train)
        print(f"naive trade-distance fit (for comparison only): A={naive.A:.4g}/s  k={naive.k:.4g}")
    except ValueError as exc:
        print(f"naive fit not possible: {exc}")

    modes = ["through", "touch"] if args.fill_mode == "both" else [args.fill_mode]
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, pair_rows = [], []
    for mode in modes:
        deltas = (np.arange(args.fit_ticks) + 0.5) * tick if args.fit_ticks else None
        fit = estimate_intensity(train, latency_ms=args.latency, fill_mode=mode, deltas=deltas)
        print(f"\n=== fill model: {mode} ===")
        print(f"window fit on train: A={fit.A:.4g}/s  k={fit.k:.4g}  ({int(fit.used.sum())} of {len(fit.used)} points used, "
              f"distances {fit.deltas[0]:.4g} to {fit.deltas[-1]:.4g})")
        for d, r in list(zip(fit.deltas, fit.rates))[:3]:
            print(f"  check: distance {d:.4g}: measured {r:.3g}/s, fitted {fit.A * np.exp(-fit.k * d):.3g}/s")

        as_mid = [AvellanedaStoikov.from_skew(x, tick, sigma, fit.k, args.horizon, args.q_max) for x in args.skew_ticks]
        as_micro = [] if args.no_micro else [
            AvellanedaStoikov.from_skew(x, tick, sigma, fit.k, args.horizon, args.q_max, center="micro")
            for x in args.skew_ticks]
        mid_as = as_mid[len(as_mid) // 2]
        # baseline: same average quote width as the middle AS strategy, but no inventory skew
        strategies = [Symmetric(0.5 * mid_as.total_spread(), args.q_max)] + as_mid + as_micro
        for s in as_mid:
            skew = s.gamma * sigma**2 * s.tau
            print(f"  {s.name}: total spread {s.total_spread():.4g} ({s.total_spread() / tick:.1f} ticks), "
                  f"reservation shift {skew:.4g} per lot ({skew / tick:.2f} ticks)")

        results, rows = [], []
        for s in strategies:
            res = run_backtest(test, s, lot_size=lot, tick=tick, maker_fee_bps=args.fee_bps,
                               fill_mode=mode, latency_ms=args.latency)
            results.append(res)
            row = {"fill": mode, **summarize(res, test)}
            rows.append(row)
        print()
        print(format_table(rows))
        all_rows += rows

        if as_micro:
            n = len(as_mid)
            pairs = []
            for j in range(n):
                c = paired_comparison(results[1 + n + j], results[1 + j])
                pairs.append({"fill": mode, "skew_ticks": args.skew_ticks[j], "pnl_AS": c["pnl_b"],
                              "pnl_AS_micro": c["pnl_a"], "micro_minus_AS": c["diff"], "t_diff": c["t_diff"]})
            print("\nmicroprice vs mid on the same data (t_diff > 2: micro significantly better, < -2: worse)")
            print(format_table(pairs))
            pair_rows.extend(pairs)

        tag = f"{name}_{mode}"
        n_mid = 1 + len(as_mid)
        plot_comparison(results[:n_mid], args.out / f"{tag}_pnl_inventory.png", title=f"{name}, test period, fill={mode}")
        if as_micro:
            plot_comparison([results[0]] + results[n_mid:], args.out / f"{tag}_pnl_inventory_micro.png",
                            title=f"{name}, microprice-centred AS, fill={mode}")
        plot_quotes(results[2], args.out / f"{tag}_quotes.png")
        plot_intensity(fit, args.out / f"{tag}_intensity.png", title=f"{name}, fill intensity on train, {mode}")

    with open(args.out / f"{name}_summary.csv", "w", newline="") as fh:
        keys = list(all_rows[0].keys())
        for r in all_rows[1:]:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    if pair_rows:
        with open(args.out / f"{name}_micro_vs_mid.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)
    print(f"\nsaved summary and figures in {args.out}/")
    print("read pnl_t_stat (Newey-West) as: |t| < 2 means the PnL could easily be luck; check 'fills' too.")
    print("all results come from one test period: significance here does not mean the effect holds on other days.")


if __name__ == "__main__":
    main()

"""Build the figures used in the README from the CSVs in results/ (run scripts/run_all.py first).

  python scripts/make_figures.py

Writes to docs/figures/ (tracked by git, unlike results/).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path("results")
OUT = Path("docs/figures")
SYMBOLS = ["solusdt", "ethusdt", "btcusdt"]
COLORS = {"solusdt": "tab:purple", "ethusdt": "tab:blue", "btcusdt": "tab:orange"}


def latency_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode in zip(axes, ["through", "touch"]):
        for sym in SYMBOLS:
            f = RES / f"{sym}_latency_sweep.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[(d.fill == mode) & ~d.strategy.str.contains("micro")]
            ax.plot(d.latency_ms, d.pnl_bps_of_volume, "o-", color=COLORS[sym], label=sym.upper())
        ax.set_title(f"fill model: {mode}")
        ax.set_xlabel("latency (ms)")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xticks([0, 10, 50, 200])
        ax.set_xticklabels(["0", "10", "50", "200"])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PnL (bps of traded notional)")
    axes[0].legend()
    fig.suptitle("Same quotes, slower execution: Avellaneda-Stoikov, skew 2 ticks/lot, test period")
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=150)
    plt.close(fig)


def skew_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for sym in SYMBOLS:
        f = RES / f"{sym}_summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.strategy.str.match(r"^AS\(skew=")].copy()
        d["skew_t"] = d.strategy.map(lambda s: float(re.search(r"skew=([\d.]+)t", s).group(1)))
        for mode, style in [("through", "o--"), ("touch", "s-")]:
            e = d[d.fill == mode].sort_values("skew_t")
            axes[0].plot(e["skew_t"], e.mean_abs_inv_lots, style, color=COLORS[sym], label=f"{sym.upper()} {mode}")
            axes[1].plot(e["skew_t"], e.pnl_bps_of_volume, style, color=COLORS[sym])
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("inventory skew (ticks per lot)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean |inventory| (lots)")
    axes[1].set_ylabel("PnL (bps of traded notional)")
    axes[0].legend(fontsize=7)
    fig.suptitle("More skew: much less inventory, but the loss per unit traded stays negative")
    fig.tight_layout()
    fig.savefig(OUT / "inventory_vs_skew.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    latency_figure()
    skew_figure()
    for name in ["btcusdt_through_intensity.png"]:
        if (RES / name).exists():
            shutil.copy(RES / name, OUT / name)
    print(f"figures written to {OUT}/")


if __name__ == "__main__":
    main()

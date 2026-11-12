"""Run every backtest and latency sweep in one go and save the full output to a log file.

Example:
  python scripts/run_all.py --data-dir C:\\mm_data
The log goes to results/run_all_log.txt (results/ is not tracked by git).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--symbols", nargs="+", default=["solusdt", "ethusdt", "btcusdt"])
    ap.add_argument("--notional", default="100")
    ap.add_argument("--fit-ticks", default="10")
    ap.add_argument("--log", type=Path, default=Path("results/run_all_log.txt"))
    args = ap.parse_args()

    common = ["--notional", args.notional, "--data-dir", args.data_dir, "--fit-ticks", args.fit_ticks]
    jobs = [["run_backtest.py", "--symbol", s, *common] for s in args.symbols]
    jobs += [["latency_sweep.py", "--symbol", s, *common] for s in args.symbols]

    args.log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.log, "w", encoding="utf-8") as log:
        for i, job in enumerate(jobs, 1):
            cmd = [sys.executable, str(HERE / job[0]), *job[1:]]
            header = f"\n{'=' * 100}\n[{i}/{len(jobs)}] python scripts/{' '.join(job)}\n{'=' * 100}\n"
            print(header, end="", flush=True)
            log.write(header)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            proc.wait()
            if proc.returncode != 0:
                msg = f"!! exited with code {proc.returncode}, continuing with the next job\n"
                print(msg, end="")
                log.write(msg)
            log.flush()
        done = f"\nall jobs finished in {(time.time() - t0) / 60:.1f} min. Log: {args.log}\n"
        print(done, end="")
        log.write(done)


if __name__ == "__main__":
    main()

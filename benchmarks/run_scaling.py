"""Scalability benchmark — measure pandapower OPF solve time across grid sizes.

Uses pandapower's built-in IEEE test cases:
  - case14   (14 buses)
  - case_ieee30 (30 buses)
  - case57   (57 buses)
  - case118  (118 buses)
  - case300  (300 buses)

Outputs a CSV consumed by the dashboard "Benchmarks" page and the README.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pandapower as pp

from gridpulse.topology import _prepare_for_opf, load_scaling_grid

CASES = [
    ("case14", 14),
    ("case_ieee30", 30),
    ("case57", 57),
    ("case118", 118),
    ("case300", 300),
]

OUT_CSV = ROOT / "benchmarks" / "results.csv"


def main():
    rows = []
    for name, target in CASES:
        net = load_scaling_grid(target)
        # Warm up
        try:
            pp.runpp(net, verbose=False)
        except Exception as e:
            print(f"[{name}] PF warmup failed: {e}")

        t0 = time.perf_counter()
        try:
            pp.runopp(net, verbose=False, suppress_warnings=True)
            converged = True
            err = None
        except Exception as e:
            converged = False
            err = str(e)
        ms = (time.perf_counter() - t0) * 1000.0

        n_buses = len(net.bus)
        n_lines = len(net.line)
        rows.append({
            "case": name,
            "buses": n_buses,
            "lines": n_lines,
            "converged": converged,
            "solve_ms": round(ms, 2),
            "error": err,
        })
        print(f"[{name}] buses={n_buses} lines={n_lines} converged={converged} time={ms:.1f} ms")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nResults saved to {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

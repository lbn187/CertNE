"""Recompute the Table-1 control-arm rows (results/m1_extra).

Reports, per arm: oracle multi-threshold fractions, tight-cert selection,
zero-BR certification, median tightness at the selected iterate, and a
full soundness assertion.

Usage: python experiments/analyze_extra_arms.py [--tag m1_extra]
"""

import argparse
import glob
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THS = (1e-2, 1e-3, 1e-4, 1e-5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="m1_extra")
    args = parser.parse_args()
    runs = {}
    for p in sorted(glob.glob(
            os.path.join(BASE_DIR, "results", args.tag, "*_seed*.json"))):
        with open(p) as f:
            r = json.load(f)
        h = r["history"]
        runs.setdefault(r["arm"], {})[r["seed"]] = dict(
            gap=np.asarray(h["nash_gap"]),
            tight=np.asarray(h["certificate_tight"]))
    for a in sorted(runs):
        seeds = sorted(runs[a])
        mins = np.array([runs[a][s]["gap"].min() for s in seeds])
        certs, gaps_at = [], []
        for s in seeds:
            j = int(np.argmin(runs[a][s]["tight"]))
            certs.append(runs[a][s]["tight"][j])
            gaps_at.append(runs[a][s]["gap"][j])
        certs, gaps_at = np.asarray(certs), np.asarray(gaps_at)
        orc = "  ".join(f"{np.mean(mins <= t):.2f}" for t in THS)
        sel = "  ".join(f"{np.mean(gaps_at <= t):.2f}" for t in THS)
        crt = "  ".join(f"{np.mean(certs <= t):.2f}" for t in THS)
        ratio = float(np.median(certs / np.maximum(gaps_at, 1e-300)))
        sound = all(np.all(runs[a][s]["tight"] >= runs[a][s]["gap"] - 1e-9)
                    for s in seeds)
        print(f"{a:16s} n={len(seeds)} oracle [{orc}] sel [{sel}] "
              f"cert [{crt}] med tight/gap {ratio:.1f}x sound={sound}")


if __name__ == "__main__":
    main()

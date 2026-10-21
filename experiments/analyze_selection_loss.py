"""Selection loss against the exact trajectory minimum, and the aligned-vs-
sequence certificate difference, computed from stored logs only (review
items M2 and M4).

Instance sets and their run directories:
  in-sample  m1_tight_seq + m1_pool_v2
  held-out   m1_heldout_seq + m1_pool_v2_heldout
  fresh      m1_fresh_seq + m1_pool_v2_fresh
Selection is the 16-arm C^1 argmin (pool16_common.select, kind="seq"); the
oracle minimum of an instance is the smallest exact gap over all arms and
checkpoints. The aligned certificate exists only in the in-sample grid
(m1_tight_seq / m1_pool_v2); there ``strict`` is the aligned column.

Writes results/pool16_paper/printout_selection_loss.txt. Reads only; no run
directory is modified.

Run: python3 experiments/analyze_selection_loss.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_m1_seq import ROOT  # noqa: E402
from pool16_common import (ALL_ARMS, DIR_HO_NEW, DIR_HO_PAPER, DIR_IN_NEW,  # noqa: E402
                           DIR_IN_PAPER, OUT_DIR, PAPER_ARMS, RES,
                           load_merged, select)

FRESH_PAPER = os.path.join(ROOT, "m1_fresh_seq")
FRESH_NEW = os.path.join(ROOT, "m1_pool_v2_fresh")

LINES = []


def P(name, value):
    line = f"{name}: {value}"
    LINES.append(line)
    print(line)


def selection_loss(runs, seeds, pool, tag):
    sel = select(runs, seeds, pool, "seq")
    g_sel = np.array([g for g, _, _ in sel])
    g_orc = np.array([min(runs[a][s]["gap"].min() for a in pool) for s in seeds])
    resolvable = g_orc > RES
    ratios = g_sel[resolvable] / g_orc[resolvable]
    n_zero_orc = int((~resolvable).sum())
    n_zero_orc_sel_not = int(np.sum((g_orc <= RES) & (g_sel > RES)))
    P(f"[{tag}] instances", f"{len(seeds)}")
    P(f"[{tag}] selection-loss ratio median, p90, max",
      f"{np.median(ratios):.4f}, {np.percentile(ratios, 90):.4f}, "
      f"{ratios.max():.4f} over {int(resolvable.sum())} instances with "
      f"oracle gap > 1e-12")
    P(f"[{tag}] instances excluded (oracle gap <= 1e-12)", f"{n_zero_orc}")
    P(f"[{tag}] oracle at floating-point zero, selection is not",
      f"{n_zero_orc_sel_not}")
    P(f"[{tag}] selected gap > oracle (ratio > 1 + 1e-9)",
      f"{int(np.sum(ratios > 1 + 1e-9))} of {int(resolvable.sum())}")


def aligned_gap(runs, seeds, pool, tag):
    """(C_aligned - C^1) / gap over all checkpoints with a resolvable gap."""
    vals = []
    for a in pool:
        for s in seeds:
            r = runs[a][s]
            keep = r["gap"] >= RES
            vals.extend(((r["strict"][keep] - r["seq"][keep]) / r["gap"][keep]).tolist())
    vals = np.asarray(vals)
    P(f"[{tag}] (aligned - C1)/gap median, p90 over checkpoints",
      f"{np.median(vals):.4f}, {np.percentile(vals, 90):.4f} over {len(vals)} "
      f"checkpoints with gap > 1e-12")
    P(f"[{tag}] aligned >= C1 checkpoint share",
      f"{100 * np.mean(vals >= -1e-12):.1f}%")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    in_runs, in_seeds = load_merged(DIR_IN_PAPER, DIR_IN_NEW)
    ho_runs, ho_seeds = load_merged(DIR_HO_PAPER, DIR_HO_NEW)
    fr_runs, fr_seeds = load_merged(FRESH_PAPER, FRESH_NEW)
    for tag, runs, seeds in (("in-sample", in_runs, in_seeds),
                             ("held-out", ho_runs, ho_seeds),
                             ("fresh", fr_runs, fr_seeds)):
        selection_loss(runs, seeds, ALL_ARMS, tag)
    aligned_gap(in_runs, in_seeds, ALL_ARMS, "in-sample 16-arm pool")
    aligned_gap(in_runs, in_seeds, PAPER_ARMS, "in-sample 7-arm pool")
    with open(os.path.join(OUT_DIR, "printout_selection_loss.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"wrote {OUT_DIR}/printout_selection_loss.txt")


if __name__ == "__main__":
    main()

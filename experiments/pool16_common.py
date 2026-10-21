"""Shared pieces of the 16-arm (7 paper arms + 9 added arms) paper analysis.

Loading reuses analyze_m1_seq.load (same record format in every directory);
selectors reuse analyze_m1_seq.portfolio_rows / ablation. Nothing here writes.
"""
import math
import os
import sys

import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
from analyze_m1_seq import ARMS as PAPER_ARMS, ROOT, load  # noqa: E402
TH = 1e-5  # the decisive threshold used throughout the text
from run_arm_pool_v2 import FULL_ARMS as NEW_ARMS  # noqa: E402
from confirm_statistics import wilson_95_ci  # noqa: E402

DIR_IN_PAPER = os.path.join(ROOT, "m1_tight_seq")
DIR_IN_NEW = os.path.join(ROOT, "m1_pool_v2")
DIR_HO_PAPER = os.path.join(ROOT, "m1_heldout_seq")
DIR_HO_NEW = os.path.join(ROOT, "m1_pool_v2_heldout")
DIR_L3_PAPER = os.path.join(ROOT, "leduc3_seq")
DIR_L3_NEW = os.path.join(ROOT, "leduc3_pool_v2")
OUT_DIR = os.path.join(ROOT, "pool16_paper")

ALL_ARMS = PAPER_ARMS + NEW_ARMS

# Paper display names (added arms keep their published names).
LABEL = {
    "rm+_alt_w1": "RM+, p=1", "rm+_alt_w2": "RM+, p=2",
    "pcfr+_alt_w1": "PRM+, p=1", "pcfr+_alt_w2": "PRM+, p=2",
    "dprm+_alt_w1_lam0.3": "DPRM+, p=1, lam=0.3",
    "dprm+_alt_w2_lam0.1": "DPRM+, p=2, lam=0.1",
    "dprm+_alt_w2_lam0.3": "DPRM+, p=2, lam=0.3",
    "dcfr_alt_w2": "DCFR (a=1.5,b=0), p=2",
    "dcfr+_alt_a1.5_w4": "DCFR+ (a=1.5), p=4",
    "pdcfr+_alt_a1.5_w5": "PDCFR+ (a=1.5), p=5",
    "hs-pcfr+_alt_g30": "HS-PCFR+ (g0=30)",
    "hs-dcfr_alt_g15": "HS-DCFR (g0=15)",
    "sprm+_alt_eta10000_w2": "Smooth PRM+ (eta=1e4), p=2",
    "stprm+_alt_eta1000_w2": "Stable PRM+ (eta=1e3), p=2",
    "exrm+_sim_eta10000_w2": "ExRM+ (eta=1e4), p=2, simult.",
    "irprm+_alt_w2": "IR-PRM+, p=2",
}

# Pool ablation rows (paper arms plus one family at a time, then all).
FAMILY_POOLS = {
    "7 original": PAPER_ARMS,
    "+ discounting (DCFR, DCFR+, PDCFR+)": PAPER_ARMS + ["dcfr_alt_w2", "dcfr+_alt_a1.5_w4",
                                                        "pdcfr+_alt_a1.5_w5"],
    "+ schedules (HS-PCFR+, HS-DCFR)": PAPER_ARMS + ["hs-pcfr+_alt_g30", "hs-dcfr_alt_g15"],
    "+ stabilised RM+ (Smooth, Stable, ExRM+)": PAPER_ARMS + ["sprm+_alt_eta10000_w2",
                                                             "stprm+_alt_eta1000_w2",
                                                             "exrm+_sim_eta10000_w2"],
    "+ IR-PRM+": PAPER_ARMS + ["irprm+_alt_w2"],
    "all 16": ALL_ARMS,
}

# Table 1 single-arm rows added for the extension.
TABLE1_NEW_ROWS = ["pdcfr+_alt_a1.5_w5", "hs-pcfr+_alt_g30", "stprm+_alt_eta1000_w2"]


def load_merged(paper_dir, new_dir, arms=ALL_ARMS):
    """Union of two run directories restricted to ``arms``; common seeds."""
    runs = {}
    for d in (paper_dir, new_dir):
        for a, v in load(d).items():
            if a in arms:
                runs.setdefault(a, {}).update(v)
    missing = [a for a in arms if a not in runs]
    if missing:
        raise SystemExit(f"arms missing under {paper_dir} / {new_dir}: {missing}")
    seeds = sorted(set.intersection(*(set(runs[a]) for a in arms)))
    return runs, seeds


def score(r, kind):
    """Per-checkpoint selection score for one run (same names as the paper)."""
    if kind == "regret":            # regret-only: max_i Rhat_i
        return r["terms"][:, :, 0].max(axis=1)
    if kind == "value":             # value-corrected: max_i [Rhat_i + Delta_i]
        return (r["terms"][:, :, 0] + r["terms"][:, :, 1]).max(axis=1)
    return r[kind]                  # seq (C^1), seq_l1 (C^0), strict (aligned), short


def select(runs, seeds, pool, kind):
    """Cross-arm argmin of ``kind``; returns (gap, score, arm) per seed."""
    out = []
    for s in seeds:
        cands = []
        for a in pool:
            sc = score(runs[a][s], kind)
            k = int(np.argmin(sc))
            cands.append((float(sc[k]), a, float(runs[a][s]["gap"][k])))
        c, a, g = min(cands, key=lambda x: (x[0], x[1]))
        out.append((g, c, a))
    return out


def solved_mask(runs, seeds, pool, kind, th=1e-5):
    return np.array([g <= th for g, _, _ in select(runs, seeds, pool, kind)])


def certified_mask(runs, seeds, pool, kind, th=1e-5):
    return np.array([c <= th for _, c, _ in select(runs, seeds, pool, kind)])


def oracle_mask(runs, seeds, pool, th=1e-5):
    return np.array([min(runs[a][s]["gap"].min() for a in pool) <= th for s in seeds])


def final_mask(runs, seeds, pool, th=1e-5):
    """Final weighted average: arm chosen by C^1 at the last checkpoint."""
    ok = []
    for s in seeds:
        a = min((runs[a][s]["seq"][-1], a) for a in pool)[1]
        ok.append(runs[a][s]["gap"][-1] <= th)
    return np.array(ok)


def random_fraction(runs, seeds, pool, th=1e-5):
    return float(np.mean([np.mean([np.mean(runs[a][s]["gap"] <= th) for a in pool])
                          for s in seeds]))


def mcnemar(left, right):
    """Exact two-sided McNemar on boolean arrays: (n_left_only, n_right_only, p)."""
    left, right = np.asarray(left, bool), np.asarray(right, bool)
    b, c = int(np.sum(left & ~right)), int(np.sum(~left & right))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return b, c, min(1.0, 2.0 * tail / 2 ** n)


def wilson(k, n):
    ci = wilson_95_ci(int(k), int(n))
    return ci.lower, ci.upper


RES = 1e-12  # floating-point resolution of the exact gap; ratios below it are noise


def ratios_all_checkpoints(runs, seeds, pool, kinds, floor=RES):
    """certificate / true gap over every checkpoint of every run in the pool
    whose exact gap is at least ``floor``; ``_excluded`` counts the rest."""
    out = {k: [] for k in kinds}
    excluded = 0
    for a in pool:
        for s in seeds:
            g = runs[a][s]["gap"]
            keep = g >= floor
            excluded += int(np.sum(~keep))
            for k in kinds:
                out[k].extend((runs[a][s][k][keep] / g[keep]).tolist())
    res = {k: np.asarray(v) for k, v in out.items()}
    res["_excluded"] = excluded
    return res


def quantile_share_tightest(ratios, key="seq", grid=200):
    """Share of quantile levels (0, 0.005, ..., 1) at which ``key`` has the
    smallest ratio of all; 200 levels is the paper's convention (98.5%)."""
    qs = np.linspace(0.0, 1.0, grid + 1)
    table = {k: np.quantile(v, qs) for k, v in ratios.items() if not k.startswith("_")}
    others = np.min(np.stack([table[k] for k in table if k != key]), axis=0)
    return float(np.mean(table[key] <= others + 1e-12))

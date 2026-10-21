"""Arm-pool extension (2026-10-01): recent RM/CFR variants on the paper's
biased-Leduc grid, logged exactly like results/m1_tight_seq (T = 2^18, 128
checkpoints, aligned strict pass, C^1/C^0 certificates), so every new arm is
directly comparable with, and poolable with, the seven paper arms.

Stages (all reuse finished logs, so they compose):
  pilot   : seeds 0-9, the whole hyperparameter grid          (~1 core-h per 10 runs)
  full    : seeds 0-63, the configurations listed in FULL_ARMS (or --arms)
  heldout : seeds 64-127, only with --prereg <file> pointing at the frozen
            v2 preregistration (refuses to run otherwise); logged without the
            aligned strict pass, exactly like results/m1_heldout_seq
  fresh   : seeds 128-191, same protocol and same requirement of a frozen
            preregistration (v3); never used by any earlier stage

Output: results/m1_pool_v2/<arm>_seed<N>.json (under the code root)
        (held-out: results/m1_pool_v2_heldout/; fresh: results/m1_pool_v2_fresh/)

Usage:
  python3 experiments/run_arm_pool_v2.py --stage pilot --workers 60
  python3 experiments/run_arm_pool_v2.py --stage full --arms pdcfr+_alt_a2.3_w5 dcfr_alt_w2
  python3 experiments/run_arm_pool_v2.py --list
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))  # experiments/
sys.path.insert(0, os.path.dirname(PROJECT))
sys.path.insert(0, PROJECT)

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import make_biased_leduc  # noqa: E402
from run_m1_seq import KEEP  # noqa: E402  (same record format as m1_tight_seq)

DATA = os.path.join(os.path.dirname(PROJECT), "results")
OUT_INSAMPLE = os.path.join(DATA, "m1_pool_v2")
OUT_HELDOUT = os.path.join(DATA, "m1_pool_v2_heldout")
OUT_FRESH = os.path.join(DATA, "m1_pool_v2_fresh")
PILOT_SEEDS = range(0, 10)
FULL_SEEDS = range(0, 64)
HELDOUT_SEEDS = range(64, 128)
FRESH_SEEDS = range(128, 192)


def _alt(**kw):
    return dict(update="alternating", **kw)


# name -> CFRConfig overrides. Names encode every hyperparameter so that logs
# are self-describing; do not rename an arm once it has produced logs.
ARMS = {}

# --- priority 1: discounting family (Brown & Sandholm 2019; Xu et al. 2024) ---
ARMS["dcfr_alt_w2"] = _alt(alg="dcfr", disc_alpha=1.5, disc_beta=0.0, weight_exp=2.0)
ARMS["lcfr_alt_w1"] = _alt(alg="dcfr", disc_alpha=1.0, disc_beta=1.0, weight_exp=1.0)
ARMS["dcfr+_alt_a1.5_w4"] = _alt(alg="dcfr+", disc_alpha=1.5, weight_exp=4.0)
ARMS["dcfr+_alt_a1.5_w2"] = _alt(alg="dcfr+", disc_alpha=1.5, weight_exp=2.0)
for a in (1.5, 2.3):
    for w in (2.0, 5.0):
        ARMS[f"pdcfr+_alt_a{a:g}_w{w:g}"] = _alt(alg="pdcfr+", disc_alpha=a, weight_exp=w)
ARMS["pdcfr+_sim_a2.3_w5"] = dict(alg="pdcfr+", update="simultaneous", disc_alpha=2.3,
                                  weight_exp=5.0)

# --- priority 2a: hyperparameter schedules (Zhang, McAleer & Sandholm 2026) ---
for g0 in (15.0, 30.0):
    ARMS[f"hs-pcfr+_alt_g{g0:g}"] = _alt(alg="pcfr+", weight_schedule="hs", hs_gamma0=g0)
    ARMS[f"hs-dcfr_alt_g{g0:g}"] = _alt(alg="dcfr", disc_schedule="hs",
                                        weight_schedule="hs", hs_gamma0=g0)

# --- priority 2b: stabilised PRM+ family (Farina et al. 2023) ---
# eta rescales the regret relative to the unit l1 floor / restart level; the
# counterfactual regrets of biased Leduc carry chance reach (~1/30), so the
# useful range sits well above the paper's matrix-game eta in {0.1, 1, 10}.
for eta in (10.0, 100.0, 1000.0, 10000.0):
    for w in (1.0, 2.0):
        ARMS[f"sprm+_alt_eta{eta:g}_w{w:g}"] = _alt(alg="sprm+", step_eta=eta, weight_exp=w)
        ARMS[f"stprm+_alt_eta{eta:g}_w{w:g}"] = _alt(alg="stprm+", step_eta=eta, weight_exp=w)
for eta in (100.0, 1000.0, 10000.0):
    ARMS[f"exrm+_sim_eta{eta:g}_w2"] = dict(alg="exrm+", update="simultaneous",
                                            step_eta=eta, weight_exp=2.0)

# --- priority 3: scale-invariant IR-PRM+ (Zhang, Anagnostides & Sandholm 2025) ---
for w in (1.0, 2.0):
    ARMS[f"irprm+_alt_w{w:g}"] = _alt(alg="irprm+", weight_exp=w)
ARMS["irprm+_sim_w2"] = dict(alg="irprm+", update="simultaneous", weight_exp=2.0)

FAMILIES = {
    "dcfr": [a for a in ARMS if a.startswith(("dcfr_", "lcfr_"))],
    "dcfr+": [a for a in ARMS if a.startswith("dcfr+_")],
    "pdcfr+": [a for a in ARMS if a.startswith("pdcfr+_")],
    "hs-pcfr+": [a for a in ARMS if a.startswith("hs-pcfr+")],
    "hs-dcfr": [a for a in ARMS if a.startswith("hs-dcfr")],
    "sprm+": [a for a in ARMS if a.startswith("sprm+")],
    "stprm+": [a for a in ARMS if a.startswith("stprm+")],
    "exrm+": [a for a in ARMS if a.startswith("exrm+")],
    "irprm+": [a for a in ARMS if a.startswith("irprm+")],
}

# Filled in after the pilot by analyze_pool_v2.py --select (one arm per
# family, rule: highest oracle fraction at 1e-5 on seeds 0-9, ties broken by
# the smaller median trajectory-minimum gap). Kept empty on purpose until
# then; --stage full without --arms refuses to run.
FULL_ARMS = [
    "dcfr_alt_w2",
    "dcfr+_alt_a1.5_w4",
    "pdcfr+_alt_a1.5_w5",
    "hs-pcfr+_alt_g30",
    "hs-dcfr_alt_g15",
    "sprm+_alt_eta10000_w2",
    "stprm+_alt_eta1000_w2",
    "exrm+_sim_eta10000_w2",
    "irprm+_alt_w2",
]


def run_task(task):
    """Identical record layout to run_m1_seq.run_task; ``strict`` toggles the
    aligned start-of-round pass (True for the in-sample stages, as in
    m1_tight_seq; False for held-out, as in m1_heldout_seq)."""
    seed, beta, name, overrides, t_max, eval_every, out_dir, strict = task
    path = os.path.join(out_dir, f"{name}_seed{seed}.json")
    if os.path.exists(path):
        return {"arm": name, "seed": seed, "skipped": True}
    efg = build_efg(make_biased_leduc(beta))
    cfg = CFRConfig(t_max=t_max, eval_every=eval_every,
                    strict_regret_logging=strict, **overrides)
    out = run_cfr(efg, cfg)
    hist = out["history"]
    rec = {
        "arm": name, "seed": seed, "beta": beta, "config": out["config"],
        "min_nash_gap": out["min_nash_gap"],
        "min_nash_gap_t": out["min_nash_gap_t"],
        "final_nash_gap": out["final"]["nash_gap"],
        "final_certificate_seq": hist["certificate_seq"][-1],
        "history": {k: hist[k] for k in KEEP if k in hist},
        "wall_seconds": out["wall_seconds"],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return rec


def build_tasks(arms, seeds, t_max, eval_every, out_dir, strict):
    tasks = []
    for seed in seeds:
        beta = float(np.random.default_rng(seed).uniform(0.0, 0.5))  # same map as run_m1_grid
        for name in arms:
            tasks.append((seed, beta, name, ARMS[name], t_max, eval_every, out_dir, strict))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pilot", "full", "heldout", "fresh"),
                    default="pilot")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="arm names (default: all for pilot, FULL_ARMS otherwise)")
    ap.add_argument("--families", nargs="+", default=None,
                    help="restrict to these families (pilot)")
    ap.add_argument("--t", type=int, default=1 << 18)
    ap.add_argument("--eval-every", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prereg", default=None,
                    help="path of the frozen v2 preregistration (required for heldout)")
    ap.add_argument("--smoke", action="store_true", help="2 seeds, T=2^12, separate dir")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for fam, names in FAMILIES.items():
            print(f"[{fam}]")
            for n in names:
                print(f"  {n:<26} {ARMS[n]}")
        print(f"{len(ARMS)} arms")
        return

    arms = args.arms
    if arms is None:
        if args.stage == "pilot":
            arms = list(ARMS)
            if args.families:
                arms = [a for f in args.families for a in FAMILIES[f]]
        else:
            arms = list(FULL_ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        sys.exit(f"unknown arms: {unknown}")
    if not arms:
        sys.exit("no arms selected: pass --arms (FULL_ARMS is filled only after the pilot)")

    if args.stage == "pilot":
        seeds, out_dir = PILOT_SEEDS, OUT_INSAMPLE
    elif args.stage == "full":
        seeds, out_dir = FULL_SEEDS, OUT_INSAMPLE
    elif args.stage == "heldout":
        if not args.prereg or not os.path.exists(args.prereg):
            sys.exit("heldout stage requires --prereg <existing frozen v2 preregistration>")
        seeds, out_dir = HELDOUT_SEEDS, OUT_HELDOUT
    else:
        if not args.prereg or not os.path.exists(args.prereg):
            sys.exit("fresh stage requires --prereg <existing frozen v3 preregistration>")
        seeds, out_dir = FRESH_SEEDS, OUT_FRESH
    if args.smoke:
        seeds, args.t, args.eval_every = range(0, 2), 1 << 12, 1 << 10
        out_dir = out_dir + "_smoke"
    os.makedirs(out_dir, exist_ok=True)

    # Aligned pass on the in-sample stages only: mirrors m1_tight_seq (True)
    # for pilot/full and m1_heldout_seq (False) for heldout/fresh.
    strict = args.stage in ("pilot", "full")
    tasks = build_tasks(arms, seeds, args.t, args.eval_every, out_dir, strict)
    todo = [t for t in tasks if not os.path.exists(
        os.path.join(out_dir, f"{t[2]}_seed{t[0]}.json"))]
    print(f"stage={args.stage} {len(arms)} arms x {len(seeds)} seeds = {len(tasks)} runs "
          f"({len(todo)} to do), t={args.t}, eval_every={args.eval_every}, "
          f"workers={args.workers} -> {out_dir}", flush=True)
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for rec in pool.imap_unordered(run_task, tasks):
            if rec.get("skipped"):
                continue
            done += 1
            print(f"[{done}/{len(todo)}] {rec['arm']} seed={rec['seed']} "
                  f"min_gap={rec['min_nash_gap']:.3e} "
                  f"seq_last={rec['final_certificate_seq']:.3e} "
                  f"({rec['wall_seconds']:.0f}s, elapsed {time.time()-t0:.0f}s)",
                  flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

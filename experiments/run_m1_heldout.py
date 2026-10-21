"""Held-out instance set: the same 7-arm grid as ``run_m1_seq.py`` on
biased-Leduc instances drawn from seeds 64--127, with the same per-instance
draw ``beta = default_rng(seed).uniform(0, 0.5)``, ``T = 2**18`` and 128
checkpoints. Records ``certificate_seq`` (C^1), ``certificate_seq_l1`` (C^0),
``certificate_seq_md`` and the per-player terms, next to the exact NashGap.

The selectors, the threshold and the migration band for this run were frozen
in a preregistration before it was launched; see the paper's held-out
appendix section.

Output: results/m1_heldout_seq/<arm>_seed<N>.json (under the code root)
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

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import make_biased_leduc  # noqa: E402
from run_m1_grid import BASE_DIR, build_arms  # noqa: E402

DEFAULT_OUT = os.path.join(BASE_DIR, "results", "m1_heldout_seq")
KEEP = ("t", "nash_gap", "player_gaps", "corr_kl", "certificate",
        "avg_regret_bound", "certificate_tight", "certificate_tight_strict",
        "avg_regret_bound_strict", "certificate_seq", "certificate_seq_l1",
        "certificate_seq_md", "certificate_seq_terms")


def run_task(task):
    seed, beta, name, overrides, t_max, eval_every, out_dir = task
    path = os.path.join(out_dir, f"{name}_seed{seed}.json")
    if os.path.exists(path):
        return {"arm": name, "seed": seed, "skipped": True}
    efg = build_efg(make_biased_leduc(beta))
    cfg = CFRConfig(t_max=t_max, eval_every=eval_every,
                    strict_regret_logging=False, **overrides)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=64)
    parser.add_argument("--n-seeds", type=int, default=64)
    parser.add_argument("--t", type=int, default=1 << 18)
    parser.add_argument("--eval-every", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.smoke:
        args.n_seeds, args.t, args.eval_every = 2, 1 << 14, 1 << 11
        args.out_dir = args.out_dir + "_smoke"

    os.makedirs(args.out_dir, exist_ok=True)
    arms = build_arms((0.1, 0.3), house=False, portfolio=True)
    tasks = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        beta = float(np.random.default_rng(seed).uniform(0.0, 0.5))
        for name, overrides in arms:
            tasks.append((seed, beta, name, overrides, args.t,
                          args.eval_every, args.out_dir))
    print(f"HELDOUT seeds {args.seed_start}-{args.seed_start+args.n_seeds-1} strict=False; {len(arms)} arms x {args.n_seeds} seeds = {len(tasks)} runs, "
          f"t={args.t}, workers={args.workers} -> {args.out_dir}", flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_task, tasks), 1):
            if rec.get("skipped"):
                continue
            print(f"[{k}/{len(tasks)}] {rec['arm']} seed={rec['seed']} "
                  f"min_gap={rec['min_nash_gap']:.3e} "
                  f"seq_last={rec['final_certificate_seq']:.3e} "
                  f"({rec['wall_seconds']:.0f}s, elapsed {time.time()-t0:.0f}s)",
                  flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

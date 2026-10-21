"""Control arms, paired with the m1_tight seeds.

The three arms:
  rm+_alt_w0    -- uniform alternating RM+  (= reference alt-CFR+ caliber,
                   clean cross-implementation anchor)
  pcfr+_alt_w0  -- uniform alternating PCFR+ (closest caliber to the
                   reference PCFR+ row; isolates the missing-weighting
                   control)
  pcfr+_sim_w2  -- simultaneous quadratic PRM+ (the simultaneous case
                   covered by Theorem 4 -> theorem-covered certification
                   column)

Same seed->beta map as run_m1_grid.py; results under results/m1_extra
(kept separate from m1_tight so the paper's full-pool definition is
unchanged).

Usage: python experiments/run_extra_arms.py [--n-seeds 64]
"""

import argparse
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from run_m1_grid import run_task, BASE_DIR  # noqa: E402

ARMS = [
    ("rm+_alt_w0", dict(alg="rm+", update="alternating", weight_exp=0.0)),
    ("pcfr+_alt_w0", dict(alg="pcfr+", update="alternating",
                          weight_exp=0.0)),
    ("pcfr+_sim_w2", dict(alg="pcfr+", update="simultaneous",
                          weight_exp=2.0)),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=64)
    parser.add_argument("--t", type=int, default=1 << 18)
    parser.add_argument("--eval-every", type=int, default=2048)
    parser.add_argument("--workers", type=int,
                        default=min(16, max(1, cpu_count() - 2)))
    parser.add_argument("--tag", default="m1_extra")
    args = parser.parse_args()

    out_dir = os.path.join(BASE_DIR, "results", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for seed in range(args.n_seeds):
        beta = float(np.random.default_rng(seed).uniform(0.0, 0.5))
        for name, overrides in ARMS:
            tasks.append((seed, beta, name, overrides, args.t,
                          args.eval_every, out_dir))
    print(f"{len(ARMS)} arms x {args.n_seeds} seeds = {len(tasks)} runs, "
          f"t={args.t}, workers={args.workers} -> {out_dir}", flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_task, tasks), 1):
            print(f"[{k}/{len(tasks)}] {rec['arm']} seed={rec['seed']} "
                  f"min_gap={rec['min_nash_gap']:.3e} "
                  f"({rec['wall_seconds']:.0f}s, "
                  f"elapsed {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

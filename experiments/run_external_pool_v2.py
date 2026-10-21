"""External-validity runs for the nine added arms (arm-pool extension v2).

Same two games, instance distributions, horizon and checkpoint grid as
``run_external.py`` (Sheriff of Nottingham and general-sum Goofspiel(5),
64 paired seeds, T = 2^16, 64 checkpoints), same record layout, so the logs
pool with ``results/ext_<game>_seq`` (the seven paper arms) one-to-one.
Arms are ``run_arm_pool_v2.FULL_ARMS`` (or ``--arms``).

Output: results/ext_<game>_pool_v2/<arm>_seed<N>.json (under the code root)
Finished runs are skipped, so relaunching resumes.

Usage:
  python3 experiments/run_external_pool_v2.py --game sheriff --smoke   # 1 arm x 1 seed, T=2^10
  python3 experiments/run_external_pool_v2.py --game sheriff --workers 60
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

PROJECT = os.path.dirname(os.path.abspath(__file__))  # experiments/
sys.path.insert(0, os.path.dirname(PROJECT))
sys.path.insert(0, PROJECT)

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from run_external import build_game, sample_instance  # noqa: E402
from run_arm_pool_v2 import ARMS, DATA, FULL_ARMS  # noqa: E402

SEEDS = range(0, 64)
T_MAX = 1 << 16
EVAL_EVERY = 1024


def run_task(task):
    """Identical record layout to run_external.run_task."""
    game, seed, params, name, overrides, t_max, eval_every, out_dir = task
    path = os.path.join(out_dir, f"{name}_seed{seed}.json")
    if os.path.exists(path):
        return {"arm": name, "seed": seed, "skipped": True}
    efg = build_efg(build_game(game, params))
    cfg = CFRConfig(t_max=t_max, eval_every=eval_every, **overrides)
    out = run_cfr(efg, cfg)
    hist = out["history"]
    rec = {
        "arm": name, "seed": seed, "game": game, "params": params,
        "config": out["config"],
        "min_nash_gap": out["min_nash_gap"],
        "min_nash_gap_t": out["min_nash_gap_t"],
        "final_nash_gap": out["final"]["nash_gap"],
        "final_corr_kl": out["final"]["corr_kl"],
        "final_certificate": out["final"]["certificate"],
        "history": {"t": hist["t"], "nash_gap": hist["nash_gap"],
                    "corr_kl": hist["corr_kl"],
                    "certificate": hist["certificate"],
                    "avg_regret_bound": hist["avg_regret_bound"],
                    "certificate_tight": hist["certificate_tight"],
                    "certificate_seq": hist["certificate_seq"],
                    "certificate_seq_l1": hist["certificate_seq_l1"],
                    "certificate_seq_md": hist["certificate_seq_md"],
                    "certificate_seq_terms": hist["certificate_seq_terms"]},
        "wall_seconds": out["wall_seconds"],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=("sheriff", "goofspiel"), required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--smoke", action="store_true",
                    help="1 arm x 1 seed at T=2^10 into results/ext_<game>_pool_v2_smoke")
    a = ap.parse_args()
    arms = a.arms or FULL_ARMS
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        sys.exit(f"unknown arms: {unknown}")
    seeds, t_max, eval_every = SEEDS, T_MAX, EVAL_EVERY
    out_dir = os.path.join(DATA, f"ext_{a.game}_pool_v2")
    if a.smoke:
        arms, seeds, t_max, eval_every = arms[:1], range(0, 1), 1 << 10, 64
        out_dir += "_smoke"
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(a.game, s, sample_instance(a.game, s), n, ARMS[n], t_max, eval_every, out_dir)
             for s in seeds for n in arms]
    print(f"{a.game}: {len(arms)} arms x {len(seeds)} seeds = {len(tasks)} runs, "
          f"t={t_max}, workers={a.workers} -> {out_dir}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_task, tasks), 1):
            if rec.get("skipped"):
                print(f"[{k}/{len(tasks)}] {rec['arm']} seed={rec['seed']} skipped", flush=True)
                continue
            print(f"[{k}/{len(tasks)}] {rec['arm']} seed={rec['seed']} "
                  f"min_gap={rec['min_nash_gap']:.3e} @t={rec['min_nash_gap_t']} "
                  f"({rec['wall_seconds']:.0f}s, elapsed {time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

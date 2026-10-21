"""Three-player Leduc for the arm-pool extension: same budget and logging as
results/leduc3_seq (T = 2^16, checkpoints every 1024, aligned strict pass,
certificate_seq / certificate_coal families), arms taken from
run_arm_pool_v2.ARMS by name.

Output: results/leduc3_pool_v2/<arm>.json (under the code root)

Usage:
  python3 experiments/run_leduc3_pool_v2.py --arms pdcfr+_alt_a2.3_w5 dcfr_alt_w2 --workers 6
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PROJECT))
sys.path.insert(0, PROJECT)

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import PokerGame, leduc_rules  # noqa: E402
from run_arm_pool_v2 import ARMS  # noqa: E402
from run_leduc3_seq import KEEP  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(PROJECT), "results", "leduc3_pool_v2")


def run_task(task):
    arm, t_max, eval_every, out_dir = task
    path = os.path.join(out_dir, f"{arm}.json")
    if os.path.exists(path):
        return {"arm": arm, "skipped": True}
    efg = build_efg(PokerGame(leduc_rules(3)))
    cfg = CFRConfig(t_max=t_max, eval_every=eval_every,
                    strict_regret_logging=True, **ARMS[arm])
    out = run_cfr(efg, cfg)
    hist = out["history"]
    rec = {
        "arm": arm, "game": "leduc3", "config": out["config"],
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--t", type=int, default=1 << 16)
    ap.add_argument("--eval-every", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        sys.exit(f"unknown arms: {unknown}")
    if args.smoke:
        args.t, args.eval_every = 64, 32
        args.out_dir = args.out_dir + "_smoke"
    os.makedirs(args.out_dir, exist_ok=True)
    tasks = [(arm, args.t, args.eval_every, args.out_dir) for arm in args.arms]
    print(f"{len(tasks)} arms, t={args.t}, workers={args.workers} -> {args.out_dir}", flush=True)
    t0 = time.time()
    with Pool(min(args.workers, len(tasks))) as pool:
        for rec in pool.imap_unordered(run_task, tasks):
            if rec.get("skipped"):
                continue
            print(f"{rec['arm']} min_gap={rec['min_nash_gap']:.3e} "
                  f"seq_last={rec['final_certificate_seq']:.3e} "
                  f"({rec['wall_seconds']:.0f}s, elapsed {time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

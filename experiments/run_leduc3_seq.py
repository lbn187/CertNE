"""Three-player Leduc with the profile-sequence certificate.

Six alternating arms (RM+/PRM+, p in {0,1,2}), T = 2^16, checkpoints every
1024 iterations, recording
``certificate_seq`` / ``certificate_seq_l1`` / ``certificate_seq_md`` and
their per-player terms next to the legacy coalition certificates, with
strict (start-of-round) logging kept for the comparison.

Output: results/leduc3_seq/<arm>.json (under the code root)
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

PROJECT = os.path.dirname(os.path.abspath(__file__))  # experiments/
sys.path.insert(0, os.path.dirname(PROJECT))

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import PokerGame, leduc_rules  # noqa: E402
ARM_CONFIGS = {
    "rm+_alt_w0": dict(alg="rm+", update="alternating", weight_exp=0.0),
    "rm+_alt_w1": dict(alg="rm+", update="alternating", weight_exp=1.0),
    "rm+_alt_w2": dict(alg="rm+", update="alternating", weight_exp=2.0),
    "pcfr+_alt_w0": dict(alg="pcfr+", update="alternating", weight_exp=0.0),
    "pcfr+_alt_w1": dict(alg="pcfr+", update="alternating", weight_exp=1.0),
    "pcfr+_alt_w2": dict(alg="pcfr+", update="alternating", weight_exp=2.0),
}
PAPER_ARMS = tuple(ARM_CONFIGS)

DEFAULT_OUT = os.path.join(os.path.dirname(PROJECT), "results", "leduc3_seq")
KEEP = ("t", "nash_gap", "player_gaps", "corr_kl", "certificate",
        "avg_regret_bound", "avg_regret_bound_strict", "certificate_coal",
        "certificate_coal_lbr", "certificate_coal_strict",
        "certificate_coal_lbr_strict", "certificate_seq",
        "certificate_seq_l1", "certificate_seq_md", "certificate_seq_terms")


def run_task(task):
    arm, t_max, eval_every, out_dir, strict = task
    path = os.path.join(out_dir, f"{arm}.json")
    if os.path.exists(path):
        return {"arm": arm, "skipped": True}
    efg = build_efg(PokerGame(leduc_rules(3)))
    cfg = CFRConfig(t_max=t_max, eval_every=eval_every,
                    strict_regret_logging=strict, **ARM_CONFIGS[arm])
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=list(PAPER_ARMS))
    parser.add_argument("--t", type=int, default=1 << 16)
    parser.add_argument("--eval-every", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-strict", action="store_true",
                        help="skip the aligned start-of-round pass (halves cost)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.smoke:
        args.t, args.eval_every = 64, 32
        args.out_dir = args.out_dir + "_smoke"
    os.makedirs(args.out_dir, exist_ok=True)
    tasks = [(arm, args.t, args.eval_every, args.out_dir, not args.no_strict)
             for arm in args.arms]
    print(f"{len(tasks)} arms, t={args.t}, workers={args.workers} -> {args.out_dir}",
          flush=True)
    t0 = time.time()
    with Pool(min(args.workers, len(tasks))) as pool:
        for rec in pool.imap_unordered(run_task, tasks):
            if rec.get("skipped"):
                continue
            print(f"{rec['arm']} min_gap={rec['min_nash_gap']:.3e} "
                  f"seq_last={rec['final_certificate_seq']:.3e} "
                  f"({rec['wall_seconds']:.0f}s, elapsed {time.time()-t0:.0f}s)",
                  flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

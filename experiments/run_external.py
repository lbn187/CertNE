"""External validity: portfolio + certificates on non-Leduc general-sum EFGs.

Two games, same protocol as results/m1_tight (64 paired seeds, portfolio
arm pool, trajectory-minimum NashGap on a 2^k-style grid, tight certificate
recorded at every checkpoint):

  sheriff   : Sheriff of Nottingham (Farina et al. 2019), instance
              distribution v ~ U(1,5), p ~ U(0.5,2), s ~ U(0.5,2),
              fixed n_max=5, b_max=2, r=2.
  goofspiel : general-sum GS(5) with evaporating tie prizes, instance
              distribution prizes = random permutation of {1..5} scaled
              elementwise by U(0.8, 1.2).

Usage:
  python experiments/run_external.py --game sheriff --pilot
  python experiments/run_external.py --game sheriff --n-seeds 64
  python experiments/run_external.py --game goofspiel --summarize-only
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import make_goofspiel, make_sheriff  # noqa: E402
from run_m1_grid import BASE_DIR, build_arms, summarize  # noqa: E402

PILOT_ARMS = ("rm+_alt_w1", "pcfr+_alt_w2", "dprm+_alt_w2_lam0.3")


def sample_instance(game: str, seed: int):
    """seed -> instance params, mirroring run_m1_grid's seeded-rng pattern."""
    rng = np.random.default_rng(seed)
    if game == "sheriff":
        return {"value": float(rng.uniform(1.0, 5.0)),
                "penalty": float(rng.uniform(0.5, 2.0)),
                "compensation": float(rng.uniform(0.5, 2.0))}
    if game == "goofspiel":
        prizes = rng.permutation(np.arange(1.0, 6.0))
        prizes = prizes * rng.uniform(0.8, 1.2, size=5)
        return {"prizes": [float(v) for v in prizes]}
    raise ValueError(game)


def build_game(game: str, params):
    if game == "sheriff":
        return make_sheriff(params["value"], params["penalty"],
                            params["compensation"], n_max=5, b_max=2,
                            n_rounds=2)
    return make_goofspiel(params["prizes"])


def run_task(task):
    game, seed, params, name, overrides, t_max, eval_every, out_dir = task
    path = os.path.join(out_dir, f"{name}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=("sheriff", "goofspiel"),
                        required=True)
    parser.add_argument("--n-seeds", type=int, default=64)
    parser.add_argument("--t", type=int, default=1 << 16)
    parser.add_argument("--eval-every", type=int, default=1024)
    parser.add_argument("--workers", type=int,
                        default=min(16, max(1, cpu_count() - 2)))
    parser.add_argument("--pilot", action="store_true",
                        help="8 seeds x 3 arms to check instance difficulty")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--out-dir", default=None,
                        help="override results/ext_<game>[_pilot]")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(BASE_DIR, "results", f"ext_{args.game}"
                                           + ("_pilot" if args.pilot else ""))
    if args.summarize_only:
        summarize(out_dir)
        return
    os.makedirs(out_dir, exist_ok=True)

    arms = build_arms((), house=False, portfolio=True)
    if args.pilot:
        args.n_seeds = 8
        arms = [(n, o) for n, o in arms if n in PILOT_ARMS]

    tasks = []
    for seed in range(args.n_seeds):
        params = sample_instance(args.game, seed)
        for name, overrides in arms:
            tasks.append((args.game, seed, params, name, overrides, args.t,
                          args.eval_every, out_dir))
    print(f"{args.game}: {len(arms)} arms x {args.n_seeds} seeds = "
          f"{len(tasks)} runs, t={args.t}, workers={args.workers} "
          f"-> {out_dir}", flush=True)

    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_task, tasks), 1):
            print(f"[{k}/{len(tasks)}] {rec['arm']} seed={rec['seed']} "
                  f"min_gap={rec['min_nash_gap']:.3e} "
                  f"@t={rec['min_nash_gap_t']} "
                  f"({rec['wall_seconds']:.0f}s, elapsed {time.time()-t0:.0f}s)",
                  flush=True)
    summarize(out_dir)


if __name__ == "__main__":
    main()

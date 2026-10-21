"""DPRM+ vs PCFR+/RM+ on the biased_2p_leduc benchmark of the NPCFR study.

Protocol: games sampled with
beta ~ U(0, 1/2), 2^18 iterations, trajectory-minimum NashGap, uniform
averaging, 'plus' algorithms with alternating updates. Success threshold
1e-5 (the NPCFR study reports PCFR+ at 0.16 and NPCFR+ at 0.96 on this
benchmark).

Stage 1 (default): 10 seeds x {baselines + DPRM+ lambda log-grid}.
Stage 2 (paper-scale): rerun with --n-seeds 64 --lams <best> for the
fraction-solved table; finished runs are skipped, so stages compose.

Usage:
  python experiments/run_m1_grid.py --smoke          # quick validation
  python experiments/run_m1_grid.py                  # full stage-1 grid
  python experiments/run_m1_grid.py --summarize-only # aggregate results/m1
"""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import make_biased_leduc  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5)
DEFAULT_LAMS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
HOUSE_LAMS = (0.1, 0.3)  # simultaneous / quadratic-weight secondary arms


def build_arms(lams, house: bool, portfolio: bool = False):
    """Arm list: (name, dict of CFRConfig overrides)."""
    if portfolio:
        # Portfolio pool (names match existing results/m1 files so
        # finished runs are reused; only missing seeds are computed).
        arms = [(f"rm+_alt_w{w:g}",
                 dict(alg="rm+", update="alternating", weight_exp=w))
                for w in (1.0, 2.0)]
        arms += [(f"pcfr+_alt_w{w:g}",
                  dict(alg="pcfr+", update="alternating", weight_exp=w))
                 for w in (1.0, 2.0)]
        arms += [(f"dprm+_alt_w2_lam{lam:g}",
                  dict(alg="dprm+", update="alternating", weight_exp=2.0,
                       lam=lam)) for lam in (0.1, 0.3)]
        arms.append(("dprm+_alt_w1_lam0.3",
                     dict(alg="dprm+", update="alternating", weight_exp=1.0,
                          lam=0.3)))
        return arms
    arms = []
    for wexp in (0.0, 1.0, 2.0):
        arms.append((f"pcfr+_alt_w{wexp:g}",
                     dict(alg="pcfr+", update="alternating", weight_exp=wexp)))
    for wexp in (0.0, 1.0):
        arms.append((f"rm+_alt_w{wexp:g}",
                     dict(alg="rm+", update="alternating", weight_exp=wexp)))
    for lam in lams:
        arms.append((f"dprm+_alt_w0_lam{lam:g}",
                     dict(alg="dprm+", update="alternating", weight_exp=0.0,
                          lam=lam)))
    if house:
        for lam in HOUSE_LAMS:
            arms.append((f"dprm+_alt_w2_lam{lam:g}",
                         dict(alg="dprm+", update="alternating",
                              weight_exp=2.0, lam=lam)))
            arms.append((f"dprm+_sim_w2_lam{lam:g}",
                         dict(alg="dprm+", update="simultaneous",
                              weight_exp=2.0, lam=lam)))
    return arms


def run_task(task):
    seed, beta, name, overrides, t_max, eval_every, out_dir = task
    path = os.path.join(out_dir, f"{name}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    efg = build_efg(make_biased_leduc(beta))
    cfg = CFRConfig(t_max=t_max, eval_every=eval_every, **overrides)
    out = run_cfr(efg, cfg)
    hist = out["history"]
    rec = {
        "arm": name, "seed": seed, "beta": beta, "config": out["config"],
        "min_nash_gap": out["min_nash_gap"],
        "min_nash_gap_t": out["min_nash_gap_t"],
        "final_nash_gap": out["final"]["nash_gap"],
        "final_corr_kl": out["final"]["corr_kl"],
        "final_certificate": out["final"]["certificate"],
        "history": {"t": hist["t"], "nash_gap": hist["nash_gap"],
                    "corr_kl": hist["corr_kl"],
                    "certificate": hist["certificate"],
                    "avg_regret_bound": hist["avg_regret_bound"],
                    **({"certificate_tight": hist["certificate_tight"]}
                       if "certificate_tight" in hist else {})},
        "wall_seconds": out["wall_seconds"],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return rec


def summarize(out_dir: str) -> None:
    recs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_seed*.json"))):
        with open(path) as f:
            recs.append(json.load(f))
    if not recs:
        print(f"no results in {out_dir}")
        return
    by_arm = {}
    for r in recs:
        by_arm.setdefault(r["arm"], []).append(r["min_nash_gap"])
    print(f"\n=== m1 summary ({out_dir}, {len(recs)} runs) ===")
    header = (f"{'arm':<26} {'n':>3} {'median min_gap':>15} "
              + " ".join(f"{'<=' + format(th, '.0e'):>8}" for th in THRESHOLDS))
    print(header)
    rows = {}
    for arm in sorted(by_arm):
        gaps = np.asarray(by_arm[arm])
        fracs = [float((gaps <= th).mean()) for th in THRESHOLDS]
        rows[arm] = (float(np.median(gaps)), fracs)
        print(f"{arm:<26} {len(gaps):>3} {np.median(gaps):>15.3e} "
              + " ".join(f"{fr:>8.2f}" for fr in fracs))
    # Best dprm arm vs best pcfr+ arm, paired on shared seeds (same sampled
    # games): an unpaired fraction comparison at small n mixes instance
    # difficulty with solver quality.
    def best(prefix):
        cands = {a: v for a, v in rows.items() if a.startswith(prefix)}
        return max(cands.items(),
                   key=lambda kv: (kv[1][1][-1], -kv[1][0])) if cands else None
    b_ours, b_base = best("dprm+"), best("pcfr+")
    if b_ours and b_base:
        import math
        (oa, (om, of)), (ba, (bm, bf)) = b_ours, b_base
        og = {r["seed"]: r["min_nash_gap"] for r in recs if r["arm"] == oa}
        bg = {r["seed"]: r["min_nash_gap"] for r in recs if r["arm"] == ba}
        shared = sorted(set(og) & set(bg))
        th = THRESHOLDS[-1]
        win = sum(og[s] <= th < bg[s] for s in shared)
        lose = sum(bg[s] <= th < og[s] for s in shared)
        nd = win + lose
        pval = (min(1.0, 2 * sum(math.comb(nd, k)
                                 for k in range(min(win, lose) + 1)) / 2 ** nd)
                if nd else 1.0)
        ratio = bm / om if om > 0 else float("inf")
        go = (win > lose and pval < 0.05) or ratio >= 5.0
        print(f"\nbest ours    : {oa}  frac<=1e-5 {of[-1]:.2f}  median {om:.3e}")
        print(f"best baseline: {ba}  frac<=1e-5 {bf[-1]:.2f}  median {bm:.3e}")
        print(f"paired on {len(shared)} seeds: ours-only {win}, "
              f"base-only {lose}, McNemar p~{pval:.3f}; "
              f"median ratio (base/ours) {ratio:.1f}x")
        print(f"m1 check: {'GO' if go else 'NO-GO'} "
              "(criterion: paired McNemar p<0.05 in our favor, "
              "or median >=5x lower)")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"rows": {a: {"median": m, "fracs": dict(
            zip(map(str, THRESHOLDS), fr))} for a, (m, fr) in rows.items()},
            "n_runs": len(recs)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--t", type=int, default=1 << 18)
    parser.add_argument("--eval-every", type=int, default=2048)
    parser.add_argument("--lams", type=float, nargs="+",
                        default=list(DEFAULT_LAMS))
    parser.add_argument("--no-house", action="store_true",
                        help="skip the simultaneous/quadratic secondary arms")
    parser.add_argument("--portfolio", action="store_true",
                        help="run the 7-arm portfolio pool instead")
    parser.add_argument("--workers", type=int,
                        default=min(16, max(1, cpu_count() - 2)))
    parser.add_argument("--tag", default="m1")
    parser.add_argument("--arm-filter", nargs="+", default=None,
                        help="only run arms whose name contains one of these")
    parser.add_argument("--smoke", action="store_true",
                        help="2 seeds, 2^13 steps, tiny grid")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(BASE_DIR, "results", args.tag)
    if args.summarize_only:
        summarize(out_dir)
        return
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke:
        args.n_seeds, args.t, args.eval_every = 2, 1 << 13, 1 << 11
        args.lams = [0.1, 0.3]

    arms = build_arms(tuple(args.lams), house=not args.no_house,
                      portfolio=args.portfolio)
    if args.arm_filter:
        arms = [(n, o) for n, o in arms
                if any(s in n for s in args.arm_filter)]
    tasks = []
    for seed in range(args.n_seeds):
        beta = float(np.random.default_rng(seed).uniform(0.0, 0.5))
        for name, overrides in arms:
            tasks.append((seed, beta, name, overrides, args.t,
                          args.eval_every, out_dir))
    print(f"{len(arms)} arms x {args.n_seeds} seeds = {len(tasks)} runs, "
          f"t={args.t}, workers={args.workers} -> {out_dir}")

    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(run_task, tasks), 1):
            print(f"[{k}/{len(tasks)}] {rec['arm']} seed={rec['seed']} "
                  f"beta={rec['beta']:.3f} min_gap={rec['min_nash_gap']:.3e} "
                  f"@t={rec['min_nash_gap_t']} "
                  f"({rec['wall_seconds']:.0f}s, elapsed {time.time()-t0:.0f}s)",
                  flush=True)
    summarize(out_dir)


if __name__ == "__main__":
    main()

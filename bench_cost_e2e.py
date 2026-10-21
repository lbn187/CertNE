#!/usr/bin/env python3
"""End-to-end cost of the checkpoint policies and of the certificate state
(tab:budget).

The micro-benchmark in ``bench_cost.py`` times one checkpoint's work in
isolation. This script instead trains the same biased-Leduc configuration
four times, changing only what a checkpoint records, and reports the total
wall-clock and peak RSS of each complete run. Each arm runs in its own
process so the peak-RSS reading is not shared.

Arms (``record_mode`` / ``certificate_state`` of :class:`src.cfr.CFRConfig`):

    bare  uninstrumented solver: no checkpoint work and no certificate
          accumulators (raw signed regret sums, evaluation-sequence terminal
          distributions, joint terminal distribution are never updated)
    none  certificate accumulators maintained, no checkpoint work
    c0    C^0 certificate, no backward pass
    c1    C^1 certificate, one signed backward pass per player
    gap   exact NashGap, one best-response pass per player
    baregap  bare solver plus the online exact-gap argmin: exact NashGap at
          every checkpoint, no certificate accumulators. The cheapest
          end-to-end implementation of the trajectory-minimum protocol.

Usage (one core per arm, arms run sequentially):

    python3 bench_cost_e2e.py                    # 2**18 iters, both grids
    python3 bench_cost_e2e.py --t 4096 --grids 256   # quick smoke

Two checkpoint grids are swept. The protocol grid (``eval_every=2048``,
128 checkpoints) is what the paper reports, and at that density the
checkpoint work is a fraction of a percent of the run, below process
start-up noise. The stress grid (``eval_every=16``, 16384 checkpoints)
makes the per-checkpoint cost measurable end to end and gives the
per-checkpoint price by dividing the delta by the checkpoint count.

Wall-clock depends on machine load; the deltas between arms are the
quantity to read. Peak RSS is the process high-water mark. ``--repeats k``
runs every arm k times in a per-repeat shuffled order (seeded) and reports
mean and min--max; the per-checkpoint and share columns come from the timed
checkpoint block, not from run differences.
"""

import argparse
import json
import random
import resource
import subprocess
import sys
import time

MODES = ("bare", "none", "c0", "c1", "gap", "baregap")


def run_one(mode: str, t_max: int, eval_every: int, beta: float,
            weight_exp: float) -> dict:
    """Run a single arm in this process; returns its measurements."""
    sys.path.insert(0, ".")
    import src.cfr as C
    from src import CFRConfig, build_efg, run_cfr
    from src.games import make_biased_leduc

    # Time the checkpoint block from outside the engine: run_cfr calls the
    # module-global _record, so wrapping it here leaves the engine untouched
    # and the numbers unchanged.
    spent = {"s": 0.0, "calls": 0}
    inner = C._record

    def timed(*a, **kw):
        t0 = time.perf_counter()
        inner(*a, **kw)
        spent["s"] += time.perf_counter() - t0
        spent["calls"] += 1

    C._record = timed
    efg = build_efg(make_biased_leduc(beta))
    cfg = CFRConfig(alg="pcfr+", update="alternating", weight_exp=weight_exp,
                    t_max=t_max, eval_every=eval_every,
                    record_mode=("none" if mode == "bare"
                                 else "gap" if mode == "baregap" else mode),
                    certificate_state=(mode not in ("bare", "baregap")))
    t0 = time.perf_counter()
    out = run_cfr(efg, cfg)
    wall = time.perf_counter() - t0
    C._record = inner
    maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    snap = out["selected_snapshot"]
    n_cp = len(out["history"]["t"])
    return {
        "mode": mode,
        "t_max": t_max,
        "checkpoints": n_cp,
        "wall_seconds": wall,
        "record_seconds": spent["s"],
        "record_per_checkpoint_ms": spent["s"] / n_cp * 1e3 if n_cp else 0.0,
        "record_share_pct": 100.0 * spent["s"] / wall if wall else 0.0,
        "ms_per_iteration": wall / t_max * 1e3,
        "maxrss_mb": maxrss_mb,
        "selected_score": None if snap is None else float(snap["score"]),
        "selected_t": None if snap is None else int(snap["t"]),
        # bit-for-bit check that the instrumentation does not touch training
        "xs_bar_sum": float(sum(float(x.sum()) for x in out["xs_bar"])),
        "xs_bar_hash": hash(tuple(float(v) for x in out["xs_bar"] for v in x[:64])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=1 << 18)
    ap.add_argument("--eval-every", type=int, default=2048)
    ap.add_argument("--grids", default="2048,16",
                    help="comma-separated eval_every values to sweep")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--weight-exp", type=float, default=2.0)
    ap.add_argument("--arm", choices=MODES, default=None,
                    help="internal: run one arm in this process")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of MODES (default: all)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat every arm this many times, shuffled order per repeat")
    ap.add_argument("--seed", type=int, default=0, help="order shuffle seed")
    args = ap.parse_args()

    if args.arm is not None:
        print("RESULT " + json.dumps(run_one(args.arm, args.t,
                                             args.eval_every, args.beta,
                                             args.weight_exp)), flush=True)
        return

    arms = tuple(a for a in (args.arms.split(",") if args.arms else MODES)
                 if a.strip())
    for a in arms:
        if a not in MODES:
            ap.error(f"unknown arm {a!r}")
    grids = [int(x) for x in args.grids.split(",") if x.strip()]
    rng = random.Random(args.seed)
    all_rows = {}
    for ee in grids:
        rows = []
        order = []
        for _ in range(args.repeats):
            perm = list(arms)
            rng.shuffle(perm)
            order.extend(perm)
        for mode in order:
            cmd = [sys.executable, __file__, "--arm", mode, "--t", str(args.t),
                   "--eval-every", str(ee), "--beta", str(args.beta),
                   "--weight-exp", str(args.weight_exp)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = [ln for ln in proc.stdout.splitlines()
                    if ln.startswith("RESULT ")]
            if not line:
                print(f"{mode}: FAILED\n{proc.stdout[-2000:]}\n"
                      f"{proc.stderr[-2000:]}", flush=True)
                continue
            rows.append(json.loads(line[-1][len("RESULT "):]))
            print(f"[eval_every={ee} {mode}] "
                  f"wall={rows[-1]['wall_seconds']:.1f}s "
                  f"peak={rows[-1]['maxrss_mb']:.0f}MB", flush=True)
        if not rows:
            continue
        all_rows[str(ee)] = rows
        print_table(aggregate(rows, arms), ee, args)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


def aggregate(rows, arms=MODES):
    """Mean and min--max per arm over repeats, in ``arms`` order."""
    out = []
    for mode in arms:
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        def stat(k):
            v = [r[k] for r in rs]
            return sum(v) / len(v), min(v), max(v)
        agg = {"mode": mode, "n": len(rs), "checkpoints": rs[0]["checkpoints"],
               "same_final_average": len({r["xs_bar_hash"] for r in rows}) == 1}
        for k in ("wall_seconds", "record_seconds", "record_share_pct",
                  "record_per_checkpoint_ms", "maxrss_mb"):
            agg[k], agg[k + "_min"], agg[k + "_max"] = stat(k)
        out.append(agg)
    return out


def print_table(rows, eval_every: int, args) -> None:
    n_cp = rows[0]["checkpoints"]
    print()
    print(f"biased Leduc beta={args.beta}, alternating PCFR+ "
          f"p={args.weight_exp:g}, T={args.t}, eval_every={eval_every} "
          f"({n_cp} checkpoints), one core per arm, {rows[0]['n']} repeat(s); "
          f"identical final average across all arms: {rows[0]['same_final_average']}")
    print(f"{'policy':<8}{'wall (s) mean [min,max]':>26}{'ckpt work (s)':>15}"
          f"{'share':>8}{'ms/ckpt':>9}{'peak RSS (MB)':>15}")
    for r in rows:
        print(f"{r['mode']:<8}"
              f"{r['wall_seconds']:>10.1f} [{r['wall_seconds_min']:.1f},{r['wall_seconds_max']:.1f}]"
              f"{r['record_seconds']:>15.2f}"
              f"{r['record_share_pct']:>7.2f}%"
              f"{r['record_per_checkpoint_ms']:>9.3f}"
              f"{r['maxrss_mb']:>15.0f}")


if __name__ == "__main__":
    main()

"""Analysis of the arm-pool extension (results/m1_pool_v2) against the paper's
seven arms (results/m1_tight_seq).

Per new arm (on the seeds it shares with the paper grid):
  oracle          trajectory-minimum exact gap <= threshold
  C1 selection    exact gap at argmin_t certificate_seq <= threshold
  C1 certification certificate_seq at its argmin <= threshold
  tightness       median certificate_seq / gap at the selected checkpoint
  soundness       #checkpoints with certificate_seq < gap - 1e-12
  union           oracle of {7 paper arms} U {arm}   (what the pool would gain)
  pool+arm        C1 portfolio selection over the 8 arms
  wall            median seconds per run
--select prints, per family, the arm the pilot rule picks (highest oracle
fraction at 1e-5, ties by smaller median min-gap); the frozen FULL_ARMS
list of run_arm_pool_v2.py records this selection.

Run:
  python3 experiments/analyze_pool_v2.py [--dir results/m1_pool_v2] [--select]
          [--pool-extra arm1 arm2 ...]   # portfolio over 7 paper arms + these
          [--paper-dir results/m1_heldout_seq]   # held-out: paper arms from there
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
from analyze_m1_seq import ARMS as PAPER_ARMS, ROOT, TH, fractions, load, portfolio_rows  # noqa: E402
from run_arm_pool_v2 import FAMILIES  # noqa: E402

POOL_DIR = os.path.join(ROOT, "m1_pool_v2")


def arm_row(runs, arm, seeds, paper):
    rs = [runs[arm][s] for s in seeds]
    mins = np.array([r["gap"].min() for r in rs])
    k = [int(np.argmin(r["seq"])) for r in rs]
    sel_gap = np.array([r["gap"][j] for r, j in zip(rs, k)])
    sel_cert = np.array([r["seq"][j] for r, j in zip(rs, k)])
    k0 = [int(np.argmin(r["seq_l1"])) for r in rs]
    sel_gap0 = np.array([r["gap"][j] for r, j in zip(rs, k0)])
    sel_cert0 = np.array([r["seq_l1"][j] for r, j in zip(rs, k0)])
    viol = int(sum(np.sum(r["seq"] < r["gap"] - 1e-12) for r in rs))
    n_ck = int(sum(len(r["gap"]) for r in rs))
    union = np.array([min(mins[j], min(paper[a][s]["gap"].min() for a in PAPER_ARMS))
                      for j, s in enumerate(seeds)])
    ratios = np.concatenate([r["seq"] / np.maximum(r["gap"], 1e-300) for r in rs])
    return {
        "n_seeds": len(seeds),
        "oracle": fractions(mins),
        "median_min_gap": float(np.median(mins)),
        "c1_selection": fractions(sel_gap),
        "c1_certification": fractions(sel_cert),
        "c0_selection": fractions(sel_gap0),
        "c0_certification": fractions(sel_cert0),
        "median_tightness_at_selected": float(np.median(sel_cert / np.maximum(sel_gap, 1e-300))),
        "median_tightness_all_checkpoints": float(np.median(ratios)),
        "soundness_violations": [viol, n_ck],
        "union_oracle_with_paper_pool": fractions(union),
        "median_wall_s": float(np.median([r["wall"] for r in rs])),
    }


def select(rows):
    picks = {}
    for fam, names in FAMILIES.items():
        cands = [(rows[n]["oracle"][-1], -rows[n]["median_min_gap"], n)
                 for n in names if n in rows]
        if cands:
            picks[fam] = max(cands)[2]
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=POOL_DIR)
    ap.add_argument("--paper-dir", default=None,
                    help="directory of the 7 paper arms (default results/m1_tight_seq; "
                         "use results/m1_heldout_seq for the held-out comparison)")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--pool-extra", nargs="*", default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    paper = load(a.paper_dir)            # default results/m1_tight_seq
    missing = [x for x in PAPER_ARMS if x not in paper]
    if missing:
        sys.exit(f"paper arms missing under {a.paper_dir or 'm1_tight_seq'}: {missing}")
    new = load(a.dir)
    if not new:
        sys.exit(f"no runs under {a.dir}")
    paper_seeds = set.intersection(*(set(paper[x]) for x in PAPER_ARMS))
    rows, seeds_used = {}, {}
    for arm in sorted(new):
        seeds = sorted(set(new[arm]) & paper_seeds)
        if not seeds:
            continue
        rows[arm] = arm_row(new, arm, seeds, paper)
        seeds_used[arm] = seeds
    fmt = lambda v: " ".join(f"{x:.2f}" for x in v)  # noqa: E731
    print(f"{'arm':<26} {'n':>3} {'oracle (1e-2..1e-5)':<22} {'C1 sel':<22} {'C1 cert':<22} "
          f"{'tight@sel':>9} {'viol':>9} {'union@1e-5':>10} {'wall s':>7}")
    for arm, r in rows.items():
        print(f"{arm:<26} {r['n_seeds']:>3} {fmt(r['oracle']):<22} {fmt(r['c1_selection']):<22} "
              f"{fmt(r['c1_certification']):<22} {r['median_tightness_at_selected']:>8.2f}x "
              f"{r['soundness_violations'][0]:>4}/{r['soundness_violations'][1]:<4} "
              f"{r['union_oracle_with_paper_pool'][-1]:>10.2f} {r['median_wall_s']:>7.0f}")
    base_seeds = sorted(paper_seeds)
    base = portfolio_rows(paper, base_seeds, PAPER_ARMS)
    print(f"\npaper pool (7 arms, {len(base_seeds)} seeds): oracle union {fmt(base['oracle_union'])}, "
          f"C1 selection {fmt(base['seq']['selection'])}, certification {fmt(base['seq']['certification'])}")
    report = {"arms": rows, "seeds": {k: v for k, v in seeds_used.items()},
              "paper_pool": {"oracle_union": base["oracle_union"],
                             "c1_selection": base["seq"]["selection"],
                             "c1_certification": base["seq"]["certification"]}}
    if a.pool_extra is not None:
        extra = a.pool_extra
        seeds = sorted(paper_seeds & set.intersection(*(set(new[x]) for x in extra)))
        merged = {**{x: paper[x] for x in PAPER_ARMS}, **{x: new[x] for x in extra}}
        p = portfolio_rows(merged, seeds, PAPER_ARMS + extra)
        print(f"\npool + {extra} ({len(seeds)} seeds): oracle union {fmt(p['oracle_union'])}, "
              f"C1 selection {fmt(p['seq']['selection'])}, certification {fmt(p['seq']['certification'])}, "
              f"tight@sel {p['seq']['median_tightness_at_selected']:.2f}x, picks {p['seq']['selected_arms']}")
        report["pool_plus_extra"] = {"arms": extra, "n_seeds": len(seeds), **{
            k: p[k] for k in ("oracle_union",)}, "c1": p["seq"], "c0": p["seq_l1"]}
    if a.select:
        picks = select(rows)
        print("\npilot picks (the selection behind FULL_ARMS of run_arm_pool_v2.py):")
        print("FULL_ARMS = " + json.dumps([picks[f] for f in FAMILIES if f in picks], indent=0))
        report["pilot_picks"] = picks
    out = a.json_out or os.path.join(a.dir, "summary_pool_v2.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Three-player Leduc table from results/leduc3_seq (and, if present, the aligned
rerun results/leduc3_strict_pcfr_w2 for comparison).

Per arm: trajectory-minimum NashGap (and its iterate), final NashGap, the
profile-sequence certificates C^1 (exact comparator) and C^0 (||w||_1) at the
final checkpoint with their ratios to the true gap, the M*D form, the C^1
argmin checkpoint's true gap, soundness over all checkpoints, and the
comparator-relaxation share of the zero-BR slack.

Run: python experiments/analyze_leduc3_seq.py [--dir DIR]
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(HERE), "results", "leduc3_seq")
ORDER = ["pcfr+_alt_w2", "pcfr+_alt_w1", "pcfr+_alt_w0",
         "rm+_alt_w2", "rm+_alt_w1", "rm+_alt_w0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT)
    a = ap.parse_args()
    runs = {}
    for p in glob.glob(os.path.join(a.dir, "*.json")):
        if os.path.basename(p).startswith("summary"):
            continue
        r = json.load(open(p))
        runs[r["arm"]] = r
    rows = {}
    print(f"{'arm':14s} {'min gap':>10s} {'@t':>6s} {'final gap':>10s} {'C1':>10s} {'ratio':>6s} "
          f"{'C0':>10s} {'ratio':>6s} {'MD':>10s} {'gap@C1min':>10s} {'sound':>7s} {'share':>6s}")
    for arm in ORDER:
        if arm not in runs:
            continue
        h = runs[arm]["history"]
        gap = np.asarray(h["nash_gap"]); c1 = np.asarray(h["certificate_seq"])
        c0 = np.asarray(h["certificate_seq_l1"]); md = np.asarray(h["certificate_seq_md"])
        terms = np.asarray(h["certificate_seq_terms"])  # [T, n, 5]
        k = int(np.argmin(gap)); j = int(np.argmin(c1))
        sound = bool(np.all(c1 >= gap - 1e-12) and np.all(c0 >= gap - 1e-12))
        # comparator-relaxation share of the zero-BR (M*D) slack at the maximizing player
        share = []
        for tt in range(len(gap)):
            i = int(np.argmax(terms[tt, :, 0] + terms[tt, :, 1] + terms[tt, :, 4]))
            slack = md[tt] - gap[tt]
            share.append((terms[tt, i, 4] - terms[tt, i, 2]) / slack if slack > 0 else np.nan)
        rows[arm] = {"min_gap": float(gap[k]), "min_t": int(h["t"][k]), "final_gap": float(gap[-1]),
                     "C1_final": float(c1[-1]), "C0_final": float(c0[-1]), "MD_final": float(md[-1]),
                     "gap_at_C1_argmin": float(gap[j]), "C1_argmin_t": int(h["t"][j]),
                     "sound": sound, "median_share_MD_minus_L": float(np.nanmedian(share)),
                     "wall_seconds": runs[arm]["wall_seconds"]}
        print(f"{arm:14s} {gap[k]:10.2e} {h['t'][k]:6d} {gap[-1]:10.2e} {c1[-1]:10.2e} {c1[-1]/gap[-1]:6.2f} "
              f"{c0[-1]:10.2e} {c0[-1]/gap[-1]:6.2f} {md[-1]:10.2e} {gap[j]:10.2e} {str(sound):>7s} "
              f"{np.nanmedian(share):6.3f}")
    strict = os.path.join(os.path.dirname(a.dir), "leduc3_strict_pcfr_w2", "pcfr+_alt_w2.json")
    if os.path.exists(strict) and "pcfr+_alt_w2" in runs:
        hs = json.load(open(strict))["history"]; h = runs["pcfr+_alt_w2"]["history"]
        gap_a = np.asarray(hs["nash_gap"], float)
        dev = float(np.abs(gap_a - np.asarray(h["nash_gap"], float)).max())
        coal = np.asarray(hs["certificate_coal"], float)
        coal_s = np.asarray(hs["certificate_coal_strict"], float)
        rel = np.abs(np.asarray(hs["avg_regret_bound_strict"], float)
                     - np.asarray(hs["avg_regret_bound"], float)) / gap_a[:, None]
        rows["aligned_pcfr+_alt_w2"] = {k: hs[k][-1] for k in hs if k.endswith("_strict")}
        rows["aligned_pcfr+_alt_w2"].update({
            "trajectory_deviation": dev,
            "md_form_identical_checkpoints": int(np.sum(coal == coal_s)),
            "n_checkpoints": int(len(coal)),
            "md_form_cross_run_max_abs_diff":
                float(np.abs(coal - np.asarray(h["certificate_seq_md"], float)).max()),
            "regret_term_rel_gap_median": float(np.median(rel)),
            "regret_term_rel_gap_max": float(rel.max()),
        })
        print("aligned rerun (final values):", rows["aligned_pcfr+_alt_w2"])
    with open(os.path.join(a.dir, "summary_seq.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

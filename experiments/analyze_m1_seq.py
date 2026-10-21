"""Analysis of the profile-sequence certificate rerun (results/m1_tight_seq).

Per arm and for the 7-arm portfolio, at the paper thresholds:
  selection      -- exact gap at the checkpoint/arm with the smallest score
  certification  -- the selected score itself is <= threshold (zero-BR report)
for scores: shortcut (certificate_tight), strict (certificate_tight_strict),
seq (exact profile-sequence certificate), seq_l1 / seq_md (zero-BR forms).
Also: soundness over all checkpoints, tightness, the size of the alignment
term L, and a regression check that trajectories equal m1_tight/m1_tight_strict.

Run: python experiments/analyze_m1_seq.py [--json-out path]
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEQ = os.path.join(ROOT, "m1_tight_seq")
TH = (1e-2, 1e-3, 1e-4, 1e-5)
ARMS = ["rm+_alt_w1", "rm+_alt_w2", "pcfr+_alt_w1", "pcfr+_alt_w2",
        "dprm+_alt_w1_lam0.3", "dprm+_alt_w2_lam0.1", "dprm+_alt_w2_lam0.3"]
SCORES = ("short", "strict", "seq", "seq_l1", "seq_md")
KEYMAP = {"short": "certificate_tight", "strict": "certificate_tight_strict",
          "seq": "certificate_seq", "seq_l1": "certificate_seq_l1",
          "seq_md": "certificate_seq_md"}
# Frozen baselines of the regression check. They are a legacy run directory and
# are not bundled (MANIFEST.md §6); when absent the check is skipped.
REF_DIRS = (os.path.join(ROOT, "m1_tight"), os.path.join(ROOT, "m1_tight_strict"))


def reference_logs_available():
    return all(os.path.isdir(d) for d in REF_DIRS)


def load(root=None):
    d = root or SEQ
    paths = sorted(glob.glob(os.path.join(d, "*_seed*.json")))
    if not paths:
        raise SystemExit(f"no run logs in {d}; run the matching run script first "
                         f"(MANIFEST.md §2)")
    runs = {}
    for path in paths:
        r = json.load(open(path))
        h = r["history"]
        rec = {"gap": np.asarray(h["nash_gap"], float), "t": np.asarray(h["t"]),
               "beta": r.get("beta"), "wall": r["wall_seconds"],
               "terms": np.asarray(h["certificate_seq_terms"], float)}
        for k, hk in KEYMAP.items():
            if hk in h:
                rec[k] = np.asarray(h[hk], float)
        if "strict" not in rec:  # external runs: no aligned pass; reuse C^1 slot
            rec["strict"] = rec["seq"]
        runs.setdefault(r["arm"], {})[int(r["seed"])] = rec
    return runs


def regression(runs, seeds):
    """Trajectories must coincide with the frozen m1_tight / m1_tight_strict logs."""
    worst = {"gap_vs_m1_tight": 0.0, "short_vs_m1_tight": 0.0,
             "strict_vs_m1_tight_strict": 0.0}
    for a in ARMS:
        for s in seeds:
            new = runs[a][s]
            old = json.load(open(os.path.join(ROOT, "m1_tight", f"{a}_seed{s}.json")))["history"]
            worst["gap_vs_m1_tight"] = max(worst["gap_vs_m1_tight"], float(
                np.abs(new["gap"] - np.asarray(old["nash_gap"])).max()))
            worst["short_vs_m1_tight"] = max(worst["short_vs_m1_tight"], float(
                np.abs(new["short"] - np.asarray(old["certificate_tight"])).max()))
            olds = json.load(open(os.path.join(ROOT, "m1_tight_strict", f"{a}_seed{s}.json")))["history"]
            worst["strict_vs_m1_tight_strict"] = max(worst["strict_vs_m1_tight_strict"], float(
                np.abs(new["strict"] - np.asarray(olds["certificate_tight_strict"])).max()))
    return worst


def fractions(v):
    v = np.asarray(v, float)
    return [float(np.mean(v <= th)) for th in TH]


def arm_rows(runs, seeds):
    out = {}
    for a in ARMS:
        row = {"oracle": fractions([runs[a][s]["gap"].min() for s in seeds])}
        for sc in SCORES:
            sel_gap, sel_cert = [], []
            for s in seeds:
                r = runs[a][s]
                k = int(np.argmin(r[sc]))
                sel_gap.append(r["gap"][k]); sel_cert.append(r[sc][k])
            row[sc] = {"selection": fractions(sel_gap),
                       "certification": fractions(sel_cert),
                       "median_tightness_at_selected": float(np.median(
                           np.asarray(sel_cert) / np.maximum(sel_gap, 1e-300)))}
        out[a] = row
    return out


def portfolio_rows(runs, seeds, pool):
    out = {"oracle_union": fractions([min(runs[a][s]["gap"].min() for a in pool)
                                      for s in seeds])}
    for sc in SCORES:
        sel_gap, sel_cert, picks = [], [], {}
        for s in seeds:
            cands = []
            for a in pool:
                r = runs[a][s]
                k = int(np.argmin(r[sc]))
                cands.append((r[sc][k], a, r["gap"][k]))
            c, a, g = min(cands, key=lambda x: (x[0], x[1]))
            sel_gap.append(g); sel_cert.append(c); picks[a] = picks.get(a, 0) + 1
        out[sc] = {"selection": fractions(sel_gap), "certification": fractions(sel_cert),
                   "median_tightness_at_selected": float(np.median(
                       np.asarray(sel_cert) / np.maximum(sel_gap, 1e-300))),
                   "selected_arms": picks,
                   "selected_gaps": [float(x) for x in sel_gap]}
    return out


def soundness_and_terms(runs, seeds):
    res = {}
    for sc in SCORES:
        bad = tot = 0
        for a in ARMS:
            for s in seeds:
                r = runs[a][s]
                bad += int(np.sum(r[sc] < r["gap"] - 1e-12)); tot += len(r["gap"])
        res[f"violations_{sc}"] = [bad, tot]
    ratios = {sc: [] for sc in SCORES}
    l_over, l1_over, tighter, tot = [], [], 0, 0
    for a in ARMS:
        for s in seeds:
            r = runs[a][s]
            g = np.maximum(r["gap"], 1e-300)
            for sc in SCORES:
                ratios[sc].extend((r[sc] / g).tolist())
            l_over.extend((np.abs(r["terms"][:, 1, 2]) / g).tolist())
            l1_over.extend((r["terms"][:, 1, 3] / g).tolist())
            tighter += int(np.sum(r["seq"] <= r["strict"])); tot += len(g)
    res["median_ratio_all_checkpoints"] = {sc: float(np.median(v)) for sc, v in ratios.items()}
    res["p90_ratio_all_checkpoints"] = {sc: float(np.percentile(v, 90)) for sc, v in ratios.items()}
    res["median_absL_over_gap_player2"] = float(np.median(l_over))
    res["p90_absL_over_gap_player2"] = float(np.percentile(l_over, 90))
    res["median_l1_over_gap_player2"] = float(np.median(l1_over))
    res["seq_le_strict_fraction"] = tighter / tot
    return res


def ablation(runs, seeds, pool=ARMS, th=1e-5):
    """Selection policies at 1e-5, per arm and cross-arm.

    score policies pick argmin_t of the score within an arm and argmin over
    arms of the picked score; 'regret' drops the value/comparator terms
    (max_i Rhat_i); 'final' takes each arm's last checkpoint and chooses the
    arm by C^1 there; 'random' is the expectation over a uniform arm and
    checkpoint draw. Reported value: exact gap at the pick <= th.
    """
    def score(r, kind):
        if kind == "regret":
            return r["terms"][:, :, 0].max(axis=1)
        return r[kind]
    kinds = ("seq", "seq_l1", "strict", "short", "regret")
    per_arm, port = {}, {}
    for a in pool:
        row = {}
        for k in kinds:
            picks = [int(np.argmin(score(runs[a][s], k))) for s in seeds]
            row[k] = float(np.mean([runs[a][s]["gap"][p] <= th for s, p in zip(seeds, picks)]))
        row["final"] = float(np.mean([runs[a][s]["gap"][-1] <= th for s in seeds]))
        row["random"] = float(np.mean([np.mean(runs[a][s]["gap"] <= th) for s in seeds]))
        row["oracle"] = float(np.mean([runs[a][s]["gap"].min() <= th for s in seeds]))
        per_arm[a] = row
    for k in kinds:
        ok = 0
        for s in seeds:
            best = min((score(runs[a][s], k).min(), a) for a in pool)
            a = best[1]
            ok += runs[a][s]["gap"][int(np.argmin(score(runs[a][s], k)))] <= th
        port[k] = ok / len(seeds)
    ok = 0
    for s in seeds:
        a = min((runs[a][s]["seq"][-1], a) for a in pool)[1]
        ok += runs[a][s]["gap"][-1] <= th
    port["final"] = ok / len(seeds)
    port["random"] = float(np.mean([np.mean([np.mean(runs[a][s]["gap"] <= th) for a in pool])
                                    for s in seeds]))
    port["oracle"] = float(np.mean([min(runs[a][s]["gap"].min() for a in pool) <= th for s in seeds]))
    # agreement between C^1 and regret-only picks
    same = tot = 0
    for a in pool:
        for s in seeds:
            same += int(np.argmin(runs[a][s]["seq"])) == int(np.argmin(score(runs[a][s], "regret")))
            tot += 1
    return {"per_arm": per_arm, "portfolio": port, "seq_vs_regret_same_pick": same / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--dir", default=SEQ)
    ap.add_argument("--no-regression", action="store_true")
    a = ap.parse_args()
    a.json_out = a.json_out or os.path.join(a.dir, "summary_seq.json")
    runs = load(a.dir)
    missing = [x for x in ARMS if x not in runs]
    if missing:
        raise SystemExit(f"missing arms in {a.dir}: {missing}")
    seeds = sorted(set.intersection(*(set(runs[x]) for x in ARMS)))
    print(f"grid: {len(ARMS)} arms x {len(seeds)} seeds")
    reg = None
    if not a.no_regression:
        if reference_logs_available():
            reg = regression(runs, seeds)
        else:
            print("regression: skipped -- the frozen reference logs "
                  "results/m1_tight and results/m1_tight_strict are not bundled "
                  "(MANIFEST.md §6)")
    rep = {"n_seeds": len(seeds), "thresholds": list(TH),
           "regression": reg,
           "soundness_terms": soundness_and_terms(runs, seeds),
           "arms": arm_rows(runs, seeds),
           "portfolio_full": portfolio_rows(runs, seeds, ARMS),
           "portfolio_no_dprm": portfolio_rows(runs, seeds, ARMS[:4]),
           "portfolio_prm_core": portfolio_rows(
               runs, seeds, ["pcfr+_alt_w1", "pcfr+_alt_w2", "dprm+_alt_w2_lam0.3"]),
           "ablation": ablation(runs, seeds),
           "wall_core_hours": float(sum(runs[x][s]["wall"] for x in ARMS for s in seeds) / 3600)}
    print("regression:", rep["regression"])
    print("soundness/terms:", json.dumps(rep["soundness_terms"], indent=1))
    fmt = lambda v: " ".join(f"{x:.2f}" for x in v)  # noqa: E731
    print(f"\n{'arm':<22} {'oracle':<20}" + "".join(f"| {sc} sel / cert{'':<14}" for sc in SCORES))
    for x in ARMS:
        row = rep["arms"][x]
        print(f"{x:<22} {fmt(row['oracle']):<20}" + "".join(
            f"| {fmt(row[sc]['selection'])} / {fmt(row[sc]['certification'])} " for sc in SCORES))
    for name in ("portfolio_full", "portfolio_no_dprm", "portfolio_prm_core"):
        p = rep[name]
        print(f"\n{name}: oracle union {fmt(p['oracle_union'])}")
        for sc in SCORES:
            print(f"  {sc:<7} selection {fmt(p[sc]['selection'])}  certification "
                  f"{fmt(p[sc]['certification'])}  tight@sel {p[sc]['median_tightness_at_selected']:.2f}x"
                  f"  arms {p[sc]['selected_arms']}")
    ab = rep["ablation"]
    print("\nablation @1e-5 (per arm: seq seq_l1 strict short regret final random oracle)")
    for x in ARMS:
        r = ab["per_arm"][x]
        print(f"  {x:<22} " + " ".join(f"{r[k]:.2f}" for k in ("seq", "seq_l1", "strict", "short", "regret", "final", "random", "oracle")))
    print("  portfolio              " + " ".join(f"{ab['portfolio'][k]:.2f}" for k in ("seq", "seq_l1", "strict", "short", "regret", "final", "random", "oracle")))
    print(f"  C1 vs regret-only same pick: {ab['seq_vs_regret_same_pick']:.3f}")
    print(f"\nwall: {rep['wall_core_hours']:.1f} core-h (with strict pass)")
    with open(a.json_out, "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", a.json_out)


if __name__ == "__main__":
    main()

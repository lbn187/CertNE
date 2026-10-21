"""Secondary-prediction readings for the frozen v3 preregistration
(experiments/PREREG_P1-3_FRESH_20261005.md, seeds 128-191).

Reads only ``results/m1_fresh_seq`` (7 paper arms) and
``results/m1_pool_v2_fresh`` (9 added arms); writes ``printout_fresh_v3.txt``
and ``summary_fresh_v3.json`` into ``results/pool16_paper`` (analysis
directory). No run directory is modified.

Readings, per prereg section 6:
  S1  S_EXACT >= S_C1                (definitional consistency)
  S2  S_C1 >= S_REG
  S3  f_cert <= f_C1                 (definitional consistency)
  S4  S_C1 and S_EXACT agree on >= 60/64 instance decisions
  S5  paired increment 16-arm vs 7-arm pool: n+ / n-, exact two-sided McNemar
  S6  added arms selected >= 55 times
  S7  point estimate and 95% interval for f_C1(fresh) - 0.921875 (held-out)
"""
import json
import math
import os
import sys

import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
from analyze_m1_seq import ARMS as PAPER_ARMS, ROOT  # noqa: E402
from pool16_common import (  # noqa: E402
    ALL_ARMS, NEW_ARMS, TH, certified_mask, load_merged, oracle_mask, select,
    solved_mask,
)
from confirm_statistics import (  # noqa: E402
    Observation, exact_mcnemar, wilson_95_ci,
)

FRESH_PAPER = os.path.join(ROOT, "m1_fresh_seq")
FRESH_NEW = os.path.join(ROOT, "m1_pool_v2_fresh")
OUT_DIR = os.path.join(ROOT, "pool16_paper")
HELDOUT_F_C1 = 59.0 / 64.0  # v2 held-out point estimate, frozen (S7 reference)

LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def n64(mask):
    return int(np.asarray(mask, dtype=bool).sum()), len(mask)


def frac64(k, n):
    return f"{k}/{n} = {k / n:.6f}"


def newcombe_diff_ci(k1, n1, k2, n2):
    """Newcombe hybrid-score 95% interval for p1 - p2 (independent groups)."""
    ci1 = wilson_95_ci(k1, n1)
    ci2 = wilson_95_ci(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - ci1.lower) ** 2 + (ci2.upper - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((ci1.upper - p1) ** 2 + (p2 - ci2.lower) ** 2)
    return lo, hi


def wald_diff_ci(k1, n1, k2, n2, z=1.959963984540054):
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) - z * se, (p1 - p2) + z * se


def main():
    runs, seeds = load_merged(FRESH_PAPER, FRESH_NEW)
    report = {"seeds": [int(seeds[0]), int(seeds[-1])], "n": len(seeds)}

    sel16 = select(runs, seeds, ALL_ARMS, "seq")
    sel7 = select(runs, seeds, PAPER_ARMS, "seq")
    gap16 = np.array([g for g, _, _ in sel16])
    cert16 = np.array([c for _, c, _ in sel16])
    arms16 = [a for _, _, a in sel16]
    gap7 = np.array([g for g, _, _ in sel7])

    solved_c1 = gap16 <= TH
    cert_c1 = cert16 <= TH
    solved_exact = oracle_mask(runs, seeds, ALL_ARMS, th=TH)
    solved_reg = solved_mask(runs, seeds, ALL_ARMS, "regret", th=TH)
    solved_7 = gap7 <= TH
    cert_c0 = certified_mask(runs, seeds, ALL_ARMS, "seq_l1", th=TH)

    emit(f"fresh seeds {seeds[0]}-{seeds[-1]}, {len(seeds)} instances, pool = "
         f"{len(ALL_ARMS)} arms, threshold {TH:g}")

    k, n = n64(solved_c1)
    emit(f"[main] f_C1 (16-arm C1 selection) = {frac64(k, n)}  "
         f"frozen band [0.81, 0.97]")
    report["f_C1_fresh"] = {"k": k, "n": n}

    ke, _ = n64(solved_exact)
    kr, _ = n64(solved_reg)
    kc1, _ = n64(cert_c1)
    kc0, _ = n64(cert_c0)
    k7, _ = n64(solved_7)
    emit(f"[S1 ] S_EXACT = {frac64(ke, n)}; S_EXACT >= S_C1: {ke >= k} "
         f"(definitional; C1-only-but-not-exact: {int((solved_c1 & ~solved_exact).sum())})")
    emit(f"[S2 ] S_REG   = {frac64(kr, n)}; S_C1 >= S_REG: {k >= kr}")
    emit(f"[S3 ] f_cert (C1) = {frac64(kc1, n)}, f_cert (C0) = {frac64(kc0, n)}; "
         f"f_cert <= f_C1: {kc1 <= k} (C1-certified-not-selected: "
         f"{int((cert_c1 & ~solved_c1).sum())})")
    agree = int((solved_c1 == solved_exact).sum())
    emit(f"[S4 ] S_C1 vs S_EXACT instance agreement: {agree}/{n} "
         f"(floor 60); disagreements (solved_C1 != solved_EXACT): {n - agree} "
         f"= {int((solved_exact & ~solved_c1).sum())} S_EXACT-only + "
         f"{int((solved_c1 & ~solved_exact).sum())} S_C1-only")
    left = [Observation(str(s), int(b)) for s, b in zip(seeds, solved_c1)]
    right = [Observation(str(s), int(b)) for s, b in zip(seeds, solved_7)]
    mc = exact_mcnemar(left, right)
    emit(f"[S5 ] paired 16-arm vs 7-arm S_C1: n+ = {mc.left_only}, "
         f"n- = {mc.right_only}, exact two-sided McNemar p = {mc.p_value:.4g}; "
         f"n+ >= n-: {mc.left_only >= mc.right_only}")
    picks = {}
    for a in arms16:
        picks[a] = picks.get(a, 0) + 1
    picks = dict(sorted(picks.items(), key=lambda kv: (-kv[1], kv[0])))
    new_picks = sum(v for a, v in picks.items() if a in NEW_ARMS)
    emit(f"[S6 ] added-arm selections = {new_picks}/{n} (floor 55); picks {picks}")

    emit()
    emit("fresh table rows (exact counts at 1e-5, then fraction):")
    table_rows = {}
    for name, pool in (("7-arm", PAPER_ARMS), ("16-arm", ALL_ARMS)):
        rows = {}
        rows["sel_C1"] = solved_mask(runs, seeds, pool, "seq", th=TH)
        rows["sel_C0"] = solved_mask(runs, seeds, pool, "seq_l1", th=TH)
        rows["sel_value"] = solved_mask(runs, seeds, pool, "value", th=TH)
        rows["sel_exact"] = oracle_mask(runs, seeds, pool, th=TH)
        rows["sel_regret"] = solved_mask(runs, seeds, pool, "regret", th=TH)
        rows["cert_C1"] = certified_mask(runs, seeds, pool, "seq", th=TH)
        rows["cert_C0"] = certified_mask(runs, seeds, pool, "seq_l1", th=TH)
        final_ok = 0
        for s in seeds:
            a = min((runs[a][s]["seq"][-1], a) for a in pool)[1]
            final_ok += runs[a][s]["gap"][-1] <= TH
        rand = float(np.mean([np.mean([np.mean(runs[a][s]["gap"] <= TH)
                                        for a in pool]) for s in seeds]))
        for key, mask in rows.items():
            kk, _ = n64(mask)
            emit(f"  [{name}] {key:<12} {frac64(kk, n)}")
            table_rows[f"{name}_{key}"] = {"k": kk, "n": n}
        emit(f"  [{name}] final        {frac64(final_ok, n)}")
        emit(f"  [{name}] random       {rand:.4f}")
        table_rows[f"{name}_final"] = {"k": final_ok, "n": n}
        table_rows[f"{name}_random"] = rand
    unsolved16 = [int(s) for s, b in zip(seeds, solved_c1) if not b]
    unsolved7 = [int(s) for s, b in zip(seeds, solved_7) if not b]
    emit(f"unsolved seeds 16-arm fresh: {unsolved16}")
    emit(f"unsolved seeds 7-arm  fresh: {unsolved7}")
    report["table_rows"] = table_rows
    report["unsolved_16"] = unsolved16
    report["unsolved_7"] = unsolved7

    diff = k / n - HELDOUT_F_C1
    lo_w, hi_w = wald_diff_ci(k, n, 59, 64)
    lo_n, hi_n = newcombe_diff_ci(k, n, 59, 64)
    emit(f"[S7 ] f_C1 fresh {frac64(k, n)} - held-out 0.921875 = {diff:+.6f}; "
         f"Wald 95% CI [{lo_w:+.4f}, {hi_w:+.4f}]; "
         f"Newcombe 95% CI [{lo_n:+.4f}, {hi_n:+.4f}] (reading only, no sided claim)")

    resid1 = min(float(np.min(runs[a][s]["seq"] - runs[a][s]["gap"]))
                 for a in ALL_ARMS for s in seeds)
    resid0 = min(float(np.min(runs[a][s]["seq_l1"] - runs[a][s]["gap"]))
                 for a in ALL_ARMS for s in seeds)
    n_ck = sum(len(runs[a][s]["gap"]) for a in ALL_ARMS for s in seeds)
    emit(f"[app] smallest C1 - gap residual over all {n_ck} checkpoints: "
         f"{resid1:.3e}; C0: {resid0:.3e} (soundness gate 1e-12)")
    if len(solved_c1):
        mm, a_m, s_m = min((TH - cert16[i], arms16[i], int(seeds[i]))
                           for i in np.where(solved_c1)[0])
        emit(f"[app] min threshold margin (TH - C1) among {int(solved_c1.sum())} "
             f"solved: {mm:.3e} on {a_m} seed {s_m}")
        report["min_margin_solved"] = {"margin": mm, "arm": a_m, "seed": s_m}
    report["min_c1_minus_gap"] = resid1
    report["min_c0_minus_gap"] = resid0

    report.update({
        "S_EXACT": {"k": ke, "n": n}, "S_REG": {"k": kr, "n": n},
        "f_cert_C1": {"k": kc1, "n": n}, "f_cert_C0": {"k": kc0, "n": n},
        "S_C1_7arm": {"k": k7, "n": n}, "S4_agreement": agree,
        "S5": {"n_plus": mc.left_only, "n_minus": mc.right_only,
               "p_value": mc.p_value},
        "S6_new_arm_picks": new_picks, "picks": picks,
        "S7": {"diff": diff, "wald": [lo_w, hi_w], "newcombe": [lo_n, hi_n]},
    })

    bad = []

    def _scan(obj, path="report"):
        if isinstance(obj, dict):
            for kk_, vv in obj.items():
                _scan(vv, f"{path}.{kk_}")
        elif isinstance(obj, (list, tuple)):
            for i, vv in enumerate(obj):
                _scan(vv, f"{path}[{i}]")
        elif hasattr(obj, "item") and not isinstance(obj, (str, bytes, bool)):
            bad.append(f"{path}:{type(obj).__name__}")

    def _py(obj):
        if isinstance(obj, dict):
            return {kk_: _py(vv) for kk_, vv in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_py(vv) for vv in obj]
        if hasattr(obj, "item") and not isinstance(obj, (str, bytes, bool)):
            return obj.item()
        return obj

    _scan(report)
    if bad:
        print("numpy scalars sanitized at: " + ", ".join(bad))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "printout_fresh_v3.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(OUT_DIR, "summary_fresh_v3.json"), "w") as fh:
        json.dump(_py(report), fh, indent=2, sort_keys=True)
    print(f"wrote {OUT_DIR}/printout_fresh_v3.txt and summary_fresh_v3.json")


if __name__ == "__main__":
    main()

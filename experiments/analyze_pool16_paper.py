"""Every number the arXiv v4 text cites for the two-player biased-Leduc grid,
recomputed from the stored logs of the 16-arm pool (7 paper arms from
results/m1_tight_seq + 9 added arms from results/m1_pool_v2; held-out from
results/m1_heldout_seq + results/m1_pool_v2_heldout).

Writes results/pool16_paper/printout_2p.txt (the number corpus for the
prose gate) and pool16_paper_2p.json. Self-check: every 7-arm figure must
reproduce the paper's current values, else exit 1.

Run: python3 experiments/analyze_pool16_paper.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool16_common import (ALL_ARMS, DIR_HO_NEW, DIR_HO_PAPER, DIR_IN_NEW,  # noqa: E402
                           DIR_IN_PAPER, FAMILY_POOLS, LABEL, NEW_ARMS, OUT_DIR, PAPER_ARMS,
                           RES, TABLE1_NEW_ROWS, TH, certified_mask, final_mask, load_merged, mcnemar,
                           oracle_mask, quantile_share_tightest, random_fraction,
                           ratios_all_checkpoints, select, solved_mask, wilson)
from analyze_m1_seq import fractions  # noqa: E402

LINES = []
REPORT = {}


def P(name, value):
    LINES.append(f"{name}: {value}")


def frac(mask):
    k, n = int(np.sum(mask)), len(mask)
    return f"{k}/{n} = {k / n:.4f} ({k / n:.2f}; {100 * k / n:.0f}%)"


def thresholds_line(values):
    """4-threshold fractions as in Table 1 (two decimals) plus counts."""
    fr = fractions(values)
    n = len(values)
    return " ".join(f"{x:.2f}" for x in fr) + "   counts " + " ".join(
        f"{int(round(x * n))}/{n}" for x in fr)


def table1(runs, seeds, pool, tag):
    P(f"[{tag}] oracle union 1e-2..1e-5",
      thresholds_line([min(runs[a][s]["gap"].min() for a in pool) for s in seeds]))
    for kind, name in (("seq", "selection C1"), ("strict", "selection aligned")):
        P(f"[{tag}] {name}", thresholds_line([g for g, _, _ in select(runs, seeds, pool, kind)]))
    for kind, name in (("seq", "certified C1"), ("seq_l1", "certified C0"),
                       ("strict", "certified aligned")):
        P(f"[{tag}] {name}", thresholds_line([c for _, c, _ in select(runs, seeds, pool, kind)]))


def tightness(runs, seeds, pool, tag, strict_available=True):
    kinds = ("seq", "seq_l1", "strict", "short") if strict_available else ("seq", "seq_l1")
    for kind in kinds:
        sel = select(runs, seeds, pool, kind)
        kept = [c / g for g, c, _ in sel if g >= RES]
        med = np.median(kept)
        P(f"[{tag}] median tightness at selected, {kind}",
          f"{med:.2f}x ({med:.1f}x) over {len(kept)} instances with selected gap >= 1e-12")
    R = ratios_all_checkpoints(runs, seeds, pool, kinds)
    P(f"[{tag}] checkpoints excluded from ratio statistics (gap < 1e-12)", f"{R['_excluded']}")
    per = {LABEL[a]: int(sum(np.sum(runs[a][s]["gap"] < RES) for s in seeds)) for a in pool}
    per = {k: v for k, v in per.items() if v}
    if per:
        P(f"[{tag}] excluded checkpoints by arm", json.dumps(per) + f"; other arms total {R['_excluded'] - max(per.values())}")
    for kind in kinds:
        P(f"[{tag}] median ratio all checkpoints, {kind}",
          f"{np.median(R[kind]):.2f}x  p90 {np.percentile(R[kind], 90):.2f}x")
    n_ck = sum(len(runs[a][s]["gap"]) for a in pool for s in seeds)
    viol = sum(int(np.sum(runs[a][s]["seq"] < runs[a][s]["gap"] - 1e-12)) for a in pool for s in seeds)
    P(f"[{tag}] checkpoints total", f"{n_ck} ({len(pool)}x{len(seeds)}x{len(runs[pool[0]][seeds[0]]['gap'])})")
    P(f"[{tag}] C1 soundness violations", f"{viol}/{n_ck}")
    if strict_available:
        P(f"[{tag}] C1 tightest share of quantile levels", f"{100 * quantile_share_tightest(R):.1f}%")
        P(f"[{tag}] C1 <= C0 share of checkpoints", f"{100 * np.mean(R['seq'] <= R['seq_l1'] + 1e-12):.1f}%")
        # comparator term sizes (player 2 under alternation): |L|/gap and ||w||_1/gap
        l_over, l1_over = [], []
        for a in pool:
            for s in seeds:
                g = np.maximum(runs[a][s]["gap"], 1e-300)
                l_over.extend((np.abs(runs[a][s]["terms"][:, 1, 2]) / g).tolist())
                l1_over.extend((runs[a][s]["terms"][:, 1, 3] / g).tolist())
        P(f"[{tag}] |L2|/gap median, p90", f"{np.median(l_over):.2f}x, {np.percentile(l_over, 90):.2f}x")
        P(f"[{tag}] ||w2||_1/gap median", f"{np.median(l1_over):.2f}x")


def soundness_residuals(runs, seeds, pool, tag):
    """Smallest computed certificate-minus-gap residual over every checkpoint
    (no exclusion; the soundness gate tolerates 1e-12)."""
    resid1 = min(float(np.min(runs[a][s]["seq"] - runs[a][s]["gap"])) for a in pool for s in seeds)
    resid0 = min(float(np.min(runs[a][s]["seq_l1"] - runs[a][s]["gap"])) for a in pool for s in seeds)
    P(f"[{tag}] smallest C1 - gap residual (all checkpoints; 1e-12 gate)", f"{resid1:.3e}")
    P(f"[{tag}] smallest C0 - gap residual (all checkpoints; 1e-12 gate)", f"{resid0:.3e}")


def margins(runs, seeds, pool, tag):
    """The margin by which ``the selected certificate is below the
    threshold on every solve'' holds. Negative margin = a solved instance
    whose C^1 is above TH."""
    sel = select(runs, seeds, pool, "seq")
    solved = [(g, c, a, s) for (g, c, a), s in zip(sel, seeds) if g <= TH]
    if not solved:
        P(f"[{tag}] min certificate margin at 1e-5", "no solved instances")
        return
    mc, a_m, s_m = min((TH - c, a, s) for _, c, a, s in solved)
    mg, g_m = min((TH - g, s) for g, _, _, s in solved)
    P(f"[{tag}] min certificate margin at 1e-5",
      f"{mc:.3e} ({100 * mc / TH:.0f}% of threshold) on {LABEL[a_m]} seed {s_m} "
      f"over {len(solved)} solved")
    P(f"[{tag}] min gap headroom at 1e-5", f"{mg:.3e} on seed {g_m}")


def failures(runs, seeds, pool, tag):
    sel = select(runs, seeds, pool, "seq")
    gaps = np.array([g for g, _, _ in sel]); certs = np.array([c for _, c, _ in sel])
    betas = np.array([runs[pool[0]][s]["beta"] for s in seeds])
    bad = gaps > TH
    P(f"[{tag}] unsolved count", f"{int(bad.sum())} of {len(seeds)}")
    P(f"[{tag}] unsolved seeds", ", ".join(str(s) for s, b in zip(seeds, bad) if b))
    P(f"[{tag}] median beta unsolved vs all", f"{np.median(betas[bad]):.2f} vs {np.median(betas):.2f}")
    P(f"[{tag}] beta>0.3 share all vs unsolved",
      f"{100 * np.mean(betas > 0.3):.0f}% vs {100 * np.mean(betas[bad] > 0.3):.0f}%")
    P(f"[{tag}] C1/gap on failures median, max",
      f"{np.median(certs[bad] / gaps[bad]):.2f}x, {np.max(certs[bad] / gaps[bad]):.2f}x")
    P(f"[{tag}] failure gaps sorted", " ".join(f"{g:.1e}" for g in np.sort(gaps[bad])))
    border = (gaps > TH) & (gaps <= 1.3e-5)
    P(f"[{tag}] borderline failures (gap <= 1.3e-5)", f"{int(border.sum())}")
    marg = (gaps <= TH) & (certs > TH)
    P(f"[{tag}] solved but uncertified count", f"{int(marg.sum())}; certificates "
      + " ".join(f"{c:.2e}" for c in np.sort(certs[marg])) + f"; max {certs[marg].max():.2e}" if marg.any() else "0")
    picks = {}
    for _, _, a in sel:
        picks[a] = picks.get(a, 0) + 1
    P(f"[{tag}] selected arms", json.dumps({LABEL[a]: picks[a] for a in sorted(picks, key=picks.get, reverse=True)}))
    P(f"[{tag}] new arms selected", f"{sum(v for a, v in picks.items() if a in NEW_ARMS)} of {len(seeds)}")


def heldout_table(runs, seeds, pool, tag, strict_available):
    rows = [("C1 selection", solved_mask(runs, seeds, pool, "seq")),
            ("C0 selection", solved_mask(runs, seeds, pool, "seq_l1")),
            ("value-corrected selection", solved_mask(runs, seeds, pool, "value")),
            ("exact-gap selection", oracle_mask(runs, seeds, pool)),
            ("regret-only selection", solved_mask(runs, seeds, pool, "regret")),
            ("certified C1", certified_mask(runs, seeds, pool, "seq")),
            ("certified C0", certified_mask(runs, seeds, pool, "seq_l1")),
            ("final average", final_mask(runs, seeds, pool))]
    if strict_available:
        rows.insert(5, ("aligned selection", solved_mask(runs, seeds, pool, "strict")))
        rows.append(("certified aligned", certified_mask(runs, seeds, pool, "strict")))
    masks = {}
    for name, m in rows:
        masks[name] = m
        lo, hi = wilson(int(m.sum()), len(m))
        P(f"[{tag}] {name}", f"{frac(m)}  Wilson [{lo:.2f}, {hi:.2f}]")
    P(f"[{tag}] uniform random checkpoint", f"{random_fraction(runs, seeds, pool):.2f}")
    P(f"[{tag}] C1 vs exact-gap agree on solved-or-not",
      f"{int(np.sum(masks['C1 selection'] == masks['exact-gap selection']))}/{len(seeds)}")
    for a, b in (("C1 selection", "regret-only selection"), ("certified C1", "certified C0"),
                 ("C1 selection", "certified C1"), ("C1 selection", "C0 selection")):
        n1, n2, p = mcnemar(masks[a], masks[b])
        P(f"[{tag}] McNemar {a} vs {b}", f"left-only {n1}, right-only {n2}, p = {p:.3f}")
    return masks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs, seeds = load_merged(DIR_IN_PAPER, DIR_IN_NEW)
    ho, ho_seeds = load_merged(DIR_HO_PAPER, DIR_HO_NEW)
    P("grid in-sample", f"{len(ALL_ARMS)} arms x {len(seeds)} seeds {seeds[0]}-{seeds[-1]}")
    P("grid held-out", f"{len(ALL_ARMS)} arms x {len(ho_seeds)} seeds {ho_seeds[0]}-{ho_seeds[-1]}")

    LINES.append("\n== Table 1 ==")
    table1(runs, seeds, PAPER_ARMS, "7 arms")
    table1(runs, seeds, ALL_ARMS, "16 arms")
    for a in TABLE1_NEW_ROWS + PAPER_ARMS:
        P(f"[single arm] {LABEL[a]} oracle", thresholds_line([runs[a][s]["gap"].min() for s in seeds]))

    LINES.append("\n== Tightness ==")
    tightness(runs, seeds, PAPER_ARMS, "7 arms")
    tightness(runs, seeds, ALL_ARMS, "16 arms")

    LINES.append("\n== Soundness residuals (all checkpoints) ==")
    soundness_residuals(runs, seeds, PAPER_ARMS, "7 arms")
    soundness_residuals(runs, seeds, ALL_ARMS, "16 arms")
    soundness_residuals(ho, ho_seeds, PAPER_ARMS, "7 arms held-out")
    soundness_residuals(ho, ho_seeds, ALL_ARMS, "16 arms held-out")

    LINES.append("\n== Certification margins at 1e-5 ==")
    margins(runs, seeds, PAPER_ARMS, "7 arms")
    margins(runs, seeds, ALL_ARMS, "16 arms")
    margins(ho, ho_seeds, PAPER_ARMS, "7 arms held-out")
    margins(ho, ho_seeds, ALL_ARMS, "16 arms held-out")

    LINES.append("\n== Failures in-sample ==")
    failures(runs, seeds, PAPER_ARMS, "7 arms")
    failures(runs, seeds, ALL_ARMS, "16 arms")
    LINES.append("\n== Failures held-out ==")
    failures(ho, ho_seeds, PAPER_ARMS, "7 arms held-out")
    failures(ho, ho_seeds, ALL_ARMS, "16 arms held-out")

    LINES.append("\n== Held-out table (in-sample columns) ==")
    m7 = heldout_table(runs, seeds, PAPER_ARMS, "7 arms", True)
    m16 = heldout_table(runs, seeds, ALL_ARMS, "16 arms", True)
    LINES.append("\n== Held-out table (held-out columns) ==")
    h7 = heldout_table(ho, ho_seeds, PAPER_ARMS, "7 arms held-out", False)
    h16 = heldout_table(ho, ho_seeds, ALL_ARMS, "16 arms held-out", False)
    LINES.append("\n== Paired increment 16 vs 7 (C1 selection) ==")
    for tag, a, b in (("in-sample", m16["C1 selection"], m7["C1 selection"]),
                      ("held-out", h16["C1 selection"], h7["C1 selection"])):
        n1, n2, p = mcnemar(a, b)
        P(f"[{tag}] 16-only, 7-only, McNemar p", f"{n1}, {n2}, p = {p:.3f}")
    for tag, a, b in (("in-sample", m16["certified C1"], m7["certified C1"]),
                      ("held-out", h16["certified C1"], h7["certified C1"])):
        n1, n2, p = mcnemar(a, b)
        P(f"[{tag}] certified C1 16-only, 7-only, McNemar p", f"{n1}, {n2}, p = {p:.3f}")
    P("[prereg v2] band", "[0.81, 0.97] from p = 0.890625 (57/64)")

    LINES.append("\n== Appendix per-arm (oracle 4 thresholds | C1 select @1e-5) ==")
    for a in ALL_ARMS:
        orc = thresholds_line([runs[a][s]["gap"].min() for s in seeds])
        sel = select(runs, seeds, [a], "seq")
        ok = sum(g <= TH for g, _, _ in sel)
        cert = sum(c <= TH for _, c, _ in sel)
        kept = [c / g for g, c, _ in sel if g >= RES]
        P(f"[per-arm] {LABEL[a]}", f"{orc} | select {ok / len(sel):.2f} ({ok}/{len(sel)}) | certified "
          f"{cert / len(sel):.2f} ({cert}/{len(sel)}) | median C1/gap at selected {np.median(kept):.2f}x over {len(kept)}")
    LINES.append("\n== Appendix selection-policy ablation per arm @1e-5 (C1 aligned regret value final random oracle) ==")
    for a in ALL_ARMS + ["POOL16", "POOL7"]:
        pool = ALL_ARMS if a == "POOL16" else PAPER_ARMS if a == "POOL7" else [a]
        vals = [np.mean(solved_mask(runs, seeds, pool, k)) for k in ("seq", "strict", "regret", "value")]
        vals += [np.mean(final_mask(runs, seeds, pool)), random_fraction(runs, seeds, pool),
                 np.mean(oracle_mask(runs, seeds, pool))]
        P(f"[ablation] {LABEL.get(a, a)}", " ".join(f"{v:.2f}" for v in vals))
    for tag, pool in (("7 arms", PAPER_ARMS), ("16 arms", ALL_ARMS)):
        same = tot = 0
        for a in pool:
            for s in seeds:
                r = runs[a][s]
                same += int(np.argmin(r["seq"])) == int(np.argmin(r["terms"][:, :, 0].max(axis=1)))
                tot += 1
        P(f"[ablation] {tag} C1 and regret-only pick the same checkpoint", f"{100 * same / tot:.0f}% of runs")
        for kind in ("regret", "strict"):
            lost = [s for s, a_, b_ in zip(seeds, solved_mask(runs, seeds, pool, "seq"), solved_mask(runs, seeds, pool, kind)) if a_ and not b_]
            P(f"[ablation] {tag} instances solved by C1 but not by {kind}", f"{len(lost)}: seeds {lost}")
    LINES.append("\n== Appendix pool ablation @1e-5 (selection C1 | certified C1 | certified C0 | aligned | oracle union) ==")
    for name, pool in FAMILY_POOLS.items():
        vals = [np.mean(solved_mask(runs, seeds, pool, "seq")), np.mean(certified_mask(runs, seeds, pool, "seq")),
                np.mean(certified_mask(runs, seeds, pool, "seq_l1")), np.mean(solved_mask(runs, seeds, pool, "strict")),
                np.mean(oracle_mask(runs, seeds, pool))]
        P(f"[pools] {name} ({len(pool)} arms)", " ".join(f"{v:.2f}" for v in vals)
          + "   counts " + " ".join(f"{int(round(v * len(seeds)))}/{len(seeds)}" for v in vals))
    LINES.append("\n== Union per threshold (16 arms): C1 selection | oracle union | aligned selection | certified C1 | certified C0 ==")
    for th in (1e-2, 1e-3, 1e-4, 1e-5):
        vals = [solved_mask(runs, seeds, ALL_ARMS, "seq", th), oracle_mask(runs, seeds, ALL_ARMS, th),
                solved_mask(runs, seeds, ALL_ARMS, "strict", th), certified_mask(runs, seeds, ALL_ARMS, "seq", th),
                certified_mask(runs, seeds, ALL_ARMS, "seq_l1", th)]
        P(f"[union] {th:g}", " ".join(str(int(v.sum())) for v in vals))

    LINES.append("\n== Budget ==")
    w_new = sum(runs[a][s]["wall"] for a in NEW_ARMS for s in seeds)
    w_old = sum(runs[a][s]["wall"] for a in PAPER_ARMS for s in seeds)
    P("[budget] 7-arm grid core-h, runs (m1_tight_seq)", f"{w_old / 3600:.1f}, {7 * len(seeds)}")
    strict_dir = os.path.join(os.path.dirname(DIR_IN_PAPER), "m1_tight_strict")
    # The aligned-grid wall-clock log is a legacy run directory; it is not
    # bundled and no script here regenerates it (MANIFEST.md §6).
    have_strict = os.path.isdir(strict_dir)
    if have_strict:
        w_strict = sum(json.load(open(os.path.join(strict_dir, f"{a}_seed{s}.json")))["wall_seconds"]
                       for a in PAPER_ARMS for s in seeds)
        P("[budget] 7-arm aligned grid core-h, runs (m1_tight_strict; the paper's 24.4)",
          f"{w_strict / 3600:.1f}, {7 * len(seeds)}")
        P("[budget] 7-arm aligned + 9 added arms (strict pass included) core-h, runs",
          f"{(w_strict + w_new) / 3600:.1f}, {16 * len(seeds)}")
    else:
        w_strict = None
        P("[budget] 7-arm aligned grid core-h, runs (m1_tight_strict; the paper's 24.4)",
          "not reproduced: reference logs not bundled (MANIFEST.md §6)")
    P("[budget] 9 added arms core-h, runs", f"{w_new / 3600:.1f}, {9 * len(seeds)}")
    P("[budget] 16-arm grid core-h, runs", f"{(w_old + w_new) / 3600:.1f}, {16 * len(seeds)}")
    P("[budget] candidate snapshots", f"{16 * 128} (16x128); 7x128 = {7 * 128}")
    for a in NEW_ARMS:
        P(f"[budget] median wall s {LABEL[a]}", f"{np.median([runs[a][s]['wall'] for s in seeds]):.0f}")

    text = "\n".join(LINES)
    with open(os.path.join(OUT_DIR, "printout_2p.txt"), "w") as f:
        f.write(text + "\n")
    print(text)

    # ---- self-check against the paper's current 7-arm numbers ----
    def line(prefix):
        return next(x for x in LINES if x.startswith(prefix))
    checks = [
        ("[7 arms] selection C1:", "1.00 1.00 0.91 0.84"), ("[7 arms] selection aligned:", "1.00 1.00 0.91 0.80"),
        ("[7 arms] certified C1:", "1.00 1.00 0.89 0.75"), ("[7 arms] certified C0:", "1.00 1.00 0.88 0.73"),
        ("[7 arms] certified aligned:", "1.00 1.00 0.88 0.69"),
        ("[7 arms] median tightness at selected, seq:", "1.8"), ("[7 arms] median tightness at selected, strict:", "2.7"),
        ("[7 arms] median ratio all checkpoints, seq:", "1.34x"), ("[7 arms] median ratio all checkpoints, strict:", "1.75x"),
        ("[7 arms] C1 tightest share of quantile levels:", "98.5%"), ("[7 arms] checkpoints total:", "57344"),
        ("[7 arms] unsolved seeds:", "5, 8, 24, 29, 34, 35, 50, 53, 60, 61"),
        ("[7 arms held-out] unsolved seeds:", "65, 70, 76, 77, 85, 90, 124, 126"),
        ("[7 arms] regret-only selection:", "53/64"), ("[7 arms] final average:", "50/64"),
        ("[7 arms held-out] C1 selection:", "56/64"), ("[7 arms held-out] certified C1:", "55/64"),
        ("[7 arms held-out] value-corrected selection:", "56/64"), ("[7 arms held-out] regret-only selection:", "54/64"),
        ("[7 arms held-out] certified C0:", "54/64"), ("[7 arms held-out] final average:", "54/64"),
        ("[7 arms] uniform random checkpoint:", "0.41"), ("[7 arms held-out] uniform random checkpoint:", "0.42"),
        ("[budget] 7-arm aligned grid core-h, runs (m1_tight_strict; the paper's 24.4):", "24.4, 448"), ("[7 arms] |L2|/gap median, p90:", "0.04x, 0.25x"),
        ("[7 arms] ||w2||_1/gap median:", "0.38x"), ("[7 arms] median beta unsolved vs all:", "0.16 vs 0.28"),
        ("[7 arms] beta>0.3 share all vs unsolved:", "45% vs 20%"),
        ("[single arm] PDCFR+ (a=1.5), p=5 oracle:", "0.66"), ("[single arm] Stable PRM+ (eta=1e3), p=2 oracle:", "0.80"),
    ]
    if not have_strict:
        checks = [c for c in checks if "m1_tight_strict" not in c[0]]
    bad = [(p, want, line(p)) for p, want in checks if want not in line(p)]
    for p, want, got in bad:
        print(f"SELF-CHECK FAIL: {p} expected '{want}' in '{got}'")
    REPORT["lines"] = LINES
    with open(os.path.join(OUT_DIR, "pool16_paper_2p.json"), "w") as f:
        json.dump(REPORT, f, indent=1)
    print(f"\nself-check: {len(checks) - len(bad)}/{len(checks)} passed; wrote {OUT_DIR}/printout_2p.txt")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

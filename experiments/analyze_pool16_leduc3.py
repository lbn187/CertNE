"""Three-player Leduc numbers for the arXiv v4 text: the six paper arms
(results/leduc3_seq) and the six added arms (results/leduc3_pool_v2), T=2^16,
64 checkpoints. Also the external-validity rows (Sheriff, Goofspiel) once
results/ext_<game>_pool_v2 is complete.

Writes results/pool16_paper/printout_3p.txt (number corpus for the prose gate).
Self-check: the six paper arms must reproduce Table 3 of the paper.

Run: python3 experiments/analyze_pool16_leduc3.py [--no-external]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool16_common import DIR_L3_NEW, DIR_L3_PAPER, LABEL, OUT_DIR, ROOT  # noqa: E402
from analyze_m1_seq import ARMS as PAPER_ARMS2P, load as load2p  # noqa: E402
from run_arm_pool_v2 import FULL_ARMS  # noqa: E402

L3_PAPER = ["pcfr+_alt_w2", "pcfr+_alt_w1", "pcfr+_alt_w0", "rm+_alt_w2", "rm+_alt_w1", "rm+_alt_w0"]
L3_NEW = ["pdcfr+_alt_a1.5_w5", "hs-pcfr+_alt_g30", "stprm+_alt_eta1000_w2", "dcfr+_alt_a1.5_w4",
          "dcfr_alt_w2", "sprm+_alt_eta10000_w2"]
L3_LABEL = {"pcfr+_alt_w2": "PRM+, p=2", "pcfr+_alt_w1": "PRM+, p=1", "pcfr+_alt_w0": "PRM+, p=0",
            "rm+_alt_w2": "RM+, p=2", "rm+_alt_w1": "RM+, p=1", "rm+_alt_w0": "RM+, p=0", **LABEL}
REPORTED_NPCFR = 1e-3
LINES = []


def P(name, value):
    LINES.append(f"{name}: {value}")


def sci(x):
    return f"{x:.2e}"


def load_l3():
    rows = {}
    for d, arms in ((DIR_L3_PAPER, L3_PAPER), (DIR_L3_NEW, L3_NEW)):
        for a in arms:
            r = json.load(open(os.path.join(d, f"{a}.json")))
            h = r["history"]
            rows[a] = {k: np.asarray(h[v], float) for k, v in
                       (("gap", "nash_gap"), ("c1", "certificate_seq"), ("c0", "certificate_seq_l1"),
                        ("md", "certificate_seq_md"), ("coal", "certificate_coal"))}
            rows[a]["t"] = np.asarray(h["t"])
            rows[a]["wall"] = r["wall_seconds"]
    return rows


def leduc3():
    rows = load_l3()
    LINES.append("== Three-player Leduc (T=2^16, 64 checkpoints) ==")
    for a in L3_PAPER + L3_NEW:
        r = rows[a]
        g, c1, c0, md, t = r["gap"], r["c1"], r["c0"], r["md"], r["t"]
        k = int(np.argmin(g))
        viol = int(np.sum(c1 < g - 1e-12)) + int(np.sum(c0 < g - 1e-12)) + int(np.sum(md < g - 1e-12))
        rat = c1 / np.maximum(g, 1e-300)
        P(f"[3p] {L3_LABEL[a]}",
          f"min gap {sci(g.min())} at t={t[k]} | at T: gap {sci(g[-1])} C1 {sci(c1[-1])} ({c1[-1] / g[-1]:.2f}x) "
          f"C0 {sci(c0[-1])} ({c0[-1] / g[-1]:.1f}x) MD {sci(md[-1])} | C1/gap over checkpoints "
          f"min {rat.min():.2f}x max {rat.max():.2f}x | viol {viol}/{3 * len(g)} | wall {r['wall']:.0f}s")
    best_g = min((rows[a]["gap"].min(), a) for a in rows)
    P("[3p] smallest trajectory minimum", f"{L3_LABEL[best_g[1]]} {sci(best_g[0])}")
    best_c = min((rows[a]["c1"].min(), a, int(np.argmin(rows[a]["c1"]))) for a in rows)
    a, k = best_c[1], best_c[2]
    P("[3p] cross-arm C1 argmin (12 arms)",
      f"{L3_LABEL[a]} at t={rows[a]['t'][k]}: gap {sci(rows[a]['gap'][k])}, C1 {sci(best_c[0])} "
      f"({best_c[0] / rows[a]['gap'][k]:.2f}x); gap is {rows[a]['gap'][k] / best_g[0]:.2f}x the pool's trajectory minimum")
    P("[3p] reported NPCFR+ 1e-3 over selected C1", f"{REPORTED_NPCFR / best_c[0]:.0f}x")
    P("[3p] reported NPCFR+ 1e-3 over pool trajectory minimum", f"{REPORTED_NPCFR / best_g[0]:.0f}x")
    old6 = min((rows[a]["c1"].min(), a) for a in L3_PAPER)
    P("[3p] cross-arm C1 argmin (6 paper arms)", f"{L3_LABEL[old6[1]]} C1 {sci(old6[0])}; reported 1e-3 over it {REPORTED_NPCFR / old6[0]:.0f}x")
    at_T = [a for a in L3_PAPER + L3_NEW if int(np.argmin(rows[a]["gap"])) == len(rows[a]["gap"]) - 1]
    P("[3p] arms attaining their minimum at T", f"{len(at_T)} of 12: " + ", ".join(L3_LABEL[a] for a in at_T))
    rng = [rows[a]["c1"][-1] / rows[a]["gap"][-1] for a in L3_PAPER + L3_NEW]
    P("[3p] C1/gap at T range over 12 arms", f"{min(rng):.2f}x - {max(rng):.1f}x")
    rng6 = [rows[a]["c1"][-1] / rows[a]["gap"][-1] for a in L3_PAPER]
    P("[3p] C1/gap at T range over 6 paper arms", f"{min(rng6):.2f}x - {max(rng6):.1f}x")
    # self-check against Table 3 of the paper
    checks = [("[3p] PRM+, p=2:", "5.50e-06"), ("[3p] PRM+, p=2:", "C1 7.65e-06 (1.39x)"),
              ("[3p] PRM+, p=2:", "MD 6.42e-05"), ("[3p] RM+, p=0:", "1.78e-03"),
              ("[3p] PRM+, p=0:", "C0 2.08e-02"), ("[3p] C1/gap at T range over 6 paper arms:", "1.39x - 8.2x")]
    bad = [(p, w) for p, w in checks if not any(x.startswith(p) and w in x for x in LINES)]
    return bad


def external(game):
    """7 paper arms (ext_<game>_seq) + 9 added arms (ext_<game>_pool_v2), C^1 certificate."""
    runs = {}
    for tag in (f"ext_{game}_seq", f"ext_{game}_pool_v2"):
        for a, v in load2p(os.path.join(ROOT, tag)).items():
            runs.setdefault(a, {}).update(v)
    pool16 = PAPER_ARMS2P + FULL_ARMS
    missing = [a for a in pool16 if a not in runs]
    if missing:
        P(f"[ext {game}] MISSING arms", ", ".join(missing))
        return
    seeds = sorted(set.intersection(*(set(runs[a]) for a in pool16)))
    LINES.append(f"\n== External validity: {game} ({len(seeds)} paired seeds) ==")
    th = 1e-5

    def pool_row(pool, tag):
        gaps, certs, picks = [], [], {}
        for s in seeds:
            cands = []
            for a in pool:
                k = int(np.argmin(runs[a][s]["seq"]))
                cands.append((float(runs[a][s]["seq"][k]), a, float(runs[a][s]["gap"][k])))
            c, a, g = min(cands, key=lambda x: (x[0], x[1]))
            gaps.append(g); certs.append(c); picks[a] = picks.get(a, 0) + 1
        gaps, certs = np.array(gaps), np.array(certs)
        orc = np.array([min(runs[a][s]["gap"].min() for a in pool) <= th for s in seeds])
        n_ck = sum(len(runs[a][s]["gap"]) for a in pool for s in seeds)
        viol = sum(int(np.sum(runs[a][s]["seq"] < runs[a][s]["gap"] - 1e-12)) for a in pool for s in seeds)
        viol0 = sum(int(np.sum(runs[a][s]["seq_l1"] < runs[a][s]["gap"] - 1e-12)) for a in pool for s in seeds)
        kept = gaps >= 1e-12
        P(f"[ext {game}] {tag} C1 selection / certified / oracle union @1e-5",
          f"{np.mean(gaps <= th):.2f} / {np.mean(certs <= th):.2f} / {orc.mean():.2f}   counts "
          f"{int((gaps <= th).sum())}/{len(seeds)} {int((certs <= th).sum())}/{len(seeds)} {int(orc.sum())}/{len(seeds)}")
        P(f"[ext {game}] {tag} median C1/gap at selected (gap>=1e-12)",
          f"{np.median(certs[kept] / gaps[kept]):.2f}x over {int(kept.sum())}" if kept.any()
          else f"n/a: all {len(seeds)} selected gaps below 1e-12")
        P(f"[ext {game}] {tag} soundness C1, C0 violations", f"{viol}/{n_ck}, {viol0}/{n_ck}")
        P(f"[ext {game}] {tag} selected arms", json.dumps({L3_LABEL[a]: picks[a] for a in sorted(picks, key=picks.get, reverse=True)}))

    pool_row(PAPER_ARMS2P, "7 arms")
    pool_row(pool16, "16 arms")
    for a in pool16:
        mins = np.array([runs[a][s]["gap"].min() for s in seeds])
        sel = [int(np.argmin(runs[a][s]["seq"])) for s in seeds]
        g = np.array([runs[a][s]["gap"][k] for s, k in zip(seeds, sel)])
        c = np.array([runs[a][s]["seq"][k] for s, k in zip(seeds, sel)])
        kept = g >= 1e-12
        P(f"[ext {game}] arm {L3_LABEL[a]} oracle / C1 select / certified @1e-5 | tight@sel",
          f"{np.mean(mins <= th):.2f} / {np.mean(g <= th):.2f} / {np.mean(c <= th):.2f} | "
          + (f"{np.median(c[kept] / g[kept]):.1f}x over {int(kept.sum())}" if kept.any()
             else f"n/a: all {len(seeds)} selected gaps below 1e-12"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-external", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    bad = leduc3()
    if not a.no_external:
        for game in ("sheriff", "goofspiel"):
            external(game)
    text = "\n".join(LINES)
    with open(os.path.join(OUT_DIR, "printout_3p.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    if not a.no_external:  # the seven-arm external rows must reproduce the paper's Table 11
        for pfx, w in (("[ext sheriff] 7 arms C1 selection / certified / oracle union @1e-5:", "1.00 / 1.00 / 1.00"),
                       ("[ext sheriff] 7 arms median C1/gap at selected (gap>=1e-12):", "11.45x"),
                       ("[ext goofspiel] 7 arms C1 selection / certified / oracle union @1e-5:", "0.56 / 0.45 / 0.58"),
                       ("[ext goofspiel] 7 arms median C1/gap at selected (gap>=1e-12):", "8.28x")):
            if not any(x.startswith(pfx) and w in x for x in LINES):
                bad.append((pfx, w))
    n_checks = 6 if a.no_external else 10
    for p, w in bad:
        print(f"SELF-CHECK FAIL: {p} expected {w!r}")
    print(f"\nself-check: {n_checks - len(bad)}/{n_checks} passed; wrote {OUT_DIR}/printout_3p.txt")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

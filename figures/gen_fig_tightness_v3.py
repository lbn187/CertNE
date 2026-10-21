#!/usr/bin/env python3
"""arXiv v4: tightness figures and hard-instances figure for the 16-arm pool
(7 paper arms from results/m1_tight_seq + 9 added arms from results/m1_pool_v2).
Derived from ../../figures/gen_fig_tightness_v2.py; same canvas sizes, fonts and
rcParams. Checkpoints whose exact gap is below 1e-12 (floating-point zero) are
excluded from ratio statistics and from the ECDF; the count is printed.

(a) PRM+ p=2 on 64 biased-Leduc instances: median and IQR over instances of
    the true NashGap, C^1, C^0, the aligned start-of-round certificate and
    the Pinsker certificate, against iteration (log-log).
(b) ECDF over all 7 x 64 x 128 checkpoints of certificate / true gap for
    C^1, C^0, the aligned certificate and the unaligned shortcut.
Hard instances (appendix): beta vs gap at the C^1-selected iterate, and
    C^1 vs gap on the unsolved instances.
"""
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Input logs are not bundled. Regenerate them with experiments/run_m1_seq.py
# and experiments/run_arm_pool_v2.py --stage full (see README.md), or point
# TBD_M1_DIR / TBD_POOL_V2_DIR at existing run directories.
DATA = os.path.join(os.path.dirname(OUT_DIR), "results")
RESULTS = os.environ.get("TBD_M1_DIR", os.path.join(DATA, "m1_tight_seq"))
RESULTS_NEW = os.environ.get("TBD_POOL_V2_DIR", os.path.join(DATA, "m1_pool_v2"))
PAPER_ARMS = ["rm+_alt_w1", "rm+_alt_w2", "pcfr+_alt_w1", "pcfr+_alt_w2",
              "dprm+_alt_w1_lam0.3", "dprm+_alt_w2_lam0.1", "dprm+_alt_w2_lam0.3"]
NEW_ARMS = ["dcfr_alt_w2", "dcfr+_alt_a1.5_w4", "pdcfr+_alt_a1.5_w5", "hs-pcfr+_alt_g30",
            "hs-dcfr_alt_g15", "sprm+_alt_eta10000_w2", "stprm+_alt_eta1000_w2",
            "exrm+_sim_eta10000_w2", "irprm+_alt_w2"]
POOL = PAPER_ARMS + NEW_ARMS
ARM = "pcfr+_alt_w2"
TH = 1e-5
RES = 1e-12  # floating-point resolution of the exact gap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 6.6, "legend.frameon": False,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
})
C = {"gap": "#264653", "seq": "#237A63", "seq_l1": "#7FB3A2", "strict": "#4F5B93",
     "short": "#C9652B", "kl": "#8C8C8C", "bad": "#C9652B", "ok": "#5E8FD6"}
LABEL = {"gap": "true NashGap", "seq": r"$C^1$ (exact comparator)",
         "seq_l1": r"$C^0$ (zero-BR)", "strict": "aligned start-of-round",
         "short": "unaligned shortcut", "kl": "Pinsker (Prop. 3)"}
KEY = {"gap": "nash_gap", "seq": "certificate_seq", "seq_l1": "certificate_seq_l1",
       "strict": "certificate_tight_strict", "short": "certificate_tight", "kl": "certificate"}


def load():
    runs = {}
    for d in (RESULTS, RESULTS_NEW):
        for p in sorted(glob.glob(os.path.join(d, "*_seed*.json"))):
            r = json.load(open(p))
            if r["arm"] not in POOL:
                continue
            h = r["history"]
            rec = {k: np.asarray(h[v], float) for k, v in KEY.items()}
            rec["t"] = np.asarray(h["t"])
            rec["beta"] = r["beta"]
            runs.setdefault(r["arm"], {})[int(r["seed"])] = rec
    missing = [a for a in POOL if a not in runs]
    assert not missing, missing
    return runs


def fig_main(runs):
    """Two separate figures: (a) curves for the appendix, (b) ECDF for the main text."""
    seeds = sorted(runs[ARM])
    t = runs[ARM][seeds[0]]["t"]
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    for key, ls in (("kl", ":"), ("strict", "-."), ("seq_l1", "--"), ("seq", "-"), ("gap", "-")):
        arr = np.array([runs[ARM][s][key] for s in seeds])
        med = np.median(arr, axis=0)
        ax.plot(t, med, color=C[key], ls=ls, label=LABEL[key], zorder=3,
                lw=1.5 if key == "gap" else 1.1)
        ax.fill_between(t, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0),
                        color=C[key], alpha=0.10, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-11, 1.0)
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("NashGap or certificate")
    ax.legend(loc="lower left", ncol=1, handlelength=1.6, fontsize=6.4, labelspacing=0.3)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.97)
    fig.savefig(os.path.join(OUT_DIR, "fig_tightness_curves.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_tightness_curves.png"))
    plt.close(fig)

    arms = sorted(runs)
    ratios = {k: [] for k in ("seq", "seq_l1", "strict", "short")}
    n_ck = excluded = 0
    for a in arms:
        for s in runs[a]:
            g = runs[a][s]["gap"]
            keep = g >= RES
            n_ck += len(g); excluded += int(np.sum(~keep))
            for k in ratios:
                ratios[k].extend((runs[a][s][k][keep] / g[keep]).tolist())
    print(f"ecdf: {len(arms)} arms, {n_ck} checkpoints, {excluded} excluded (gap < 1e-12)")
    fig, bx = plt.subplots(figsize=(2.35, 1.95))
    for k, ls in (("short", ":"), ("strict", "-."), ("seq_l1", "--"), ("seq", "-")):
        v = np.sort(np.asarray(ratios[k]))
        bx.plot(v, np.arange(1, len(v) + 1) / len(v), color=C[k], ls=ls, label=LABEL[k])
    bx.axvline(1.0, color="#444", lw=0.6, ls=":")
    bx.set_xscale("log")
    bx.set_xlim(0.9, 1e3)
    bx.set_ylim(0, 1)
    bx.set_xlabel("certificate / true NashGap")
    bx.set_ylabel("fraction of checkpoints")
    bx.legend(loc="lower right", handlelength=1.5, fontsize=6.0, labelspacing=0.3)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.22, top=0.97)
    fig.savefig(os.path.join(OUT_DIR, "fig_tightness.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_tightness.png"))
    plt.close(fig)
    for k in ratios:
        v = np.asarray(ratios[k])
        print(f"{k:7s} median {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} "
              f"min {v.min():.4f} frac<1 {(v < 1 - 1e-9).mean():.4f}")


def fig_hard(runs):
    arms = sorted(runs)
    seeds = sorted(set.intersection(*(set(v) for v in runs.values())))
    print(f"hard: pool of {len(arms)} arms, {len(seeds)} seeds")
    betas, gaps, certs = [], [], []
    for s in seeds:
        cands = []
        for a in arms:
            r = runs[a][s]
            j = int(np.argmin(r["seq"]))
            cands.append((float(r["seq"][j]), a, float(r["gap"][j])))
        c, _, g = min(cands, key=lambda x: (x[0], x[1]))
        betas.append(runs[arms[0]][s]["beta"]); gaps.append(g); certs.append(c)
    betas, gaps, certs = map(np.asarray, (betas, gaps, certs))
    bad = gaps > TH
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.0),
                                   gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.35})
    ax1.scatter(betas[~bad], gaps[~bad], s=9, color=C["ok"], alpha=0.8, lw=0, label="solved")
    ax1.scatter(betas[bad], gaps[bad], s=13, color=C["bad"], marker="D", lw=0, label="unsolved")
    ax1.axhline(TH, color="#444", lw=0.6, ls=":")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"tie-refund fraction $\beta$")
    ax1.set_ylabel(r"NashGap at $C^1$-selected iterate")
    ax1.set_title("(a) hard instances", loc="left")
    ax1.legend(loc="lower left", markerscale=1.4)
    lo = min(gaps[bad].min(), certs[bad].min()) * 0.6
    hi = max(gaps[bad].max(), certs[bad].max()) * 1.7
    xs = np.array([lo, hi])
    ax2.plot(xs, xs, color="#444", lw=0.6, ls=":")
    ax2.plot(xs, 2 * xs, color="#999", lw=0.6, ls=":")
    ax2.text(hi * 0.5, hi * 0.42, r"$1\times$", fontsize=6.5, color="#444", va="top")
    ax2.text(lo * 1.15, lo * 3.6, r"$2\times$", fontsize=6.5, color="#777", va="bottom")
    ax2.scatter(gaps[bad], certs[bad], s=13, color=C["bad"], marker="D", lw=0)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_xlabel("true NashGap")
    ax2.set_ylabel(r"$C^1$")
    ax2.set_title("(b) certificate on failures", loc="left")
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.245, top=0.88)
    fig.savefig(os.path.join(OUT_DIR, "fig_hard_instances.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_hard_instances.png"))
    plt.close(fig)
    print(f"hard: {bad.sum()}/{len(seeds)} unsolved; median beta unsolved "
          f"{np.median(betas[bad]):.3f} vs all {np.median(betas):.3f}; "
          f"beta>0.3 share all {np.mean(betas > 0.3):.2f} unsolved {np.mean(betas[bad] > 0.3):.2f}; "
          f"ratio on failures median {np.median(certs[bad] / gaps[bad]):.2f} "
          f"max {np.max(certs[bad] / gaps[bad]):.2f}; unsolved gaps {np.sort(gaps[bad])}")


if __name__ == "__main__":
    runs = load()
    fig_main(runs)
    fig_hard(runs)

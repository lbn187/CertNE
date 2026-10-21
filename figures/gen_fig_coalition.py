#!/usr/bin/env python3
"""Appendix figure: three-player Leduc, PRM+ p=2 (alternating), from
results/leduc3_seq. Per-checkpoint curves of the true NashGap and the three
forms of the profile-sequence certificate: the M*D relaxation, the zero-BR
C^0 (||w||_1) and the exact-comparator C^1 (L). The vertical distance between
the M*D curve and C^1 is the comparator relaxation; the distance between C^1
and the true gap is regret slack plus the proposition's own residual.
Drawn at final size (4.2 x 2.0 in, width=0.76\\textwidth).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Input log is not bundled. Regenerate it with
# experiments/run_leduc3_seq.py --no-strict (see README.md), or point
# TBD_LEDUC3_JSON at an existing run log.
SRC = os.environ.get(
    "TBD_LEDUC3_JSON",
    os.path.join(OUT_DIR, "..", "results", "leduc3_seq",
                 "pcfr+_alt_w2.json"))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.6,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
})
C = {"gap": "#264653", "seq": "#237A63", "seq_l1": "#7FB3A2", "md": "#8C8C8C"}


def main() -> None:
    h = json.load(open(SRC))["history"]
    t = np.asarray(h["t"])
    gap = np.asarray(h["nash_gap"])
    md = np.asarray(h["certificate_seq_md"])
    c0 = np.asarray(h["certificate_seq_l1"])
    c1 = np.asarray(h["certificate_seq"])
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    for vals, key, label, ls in ((md, "md", r"$M D_{-i}$ form", ":"),
                                 (c0, "seq_l1", r"$C^0$ ($\|w_i\|_1$)", "--"),
                                 (c1, "seq", r"$C^1$ (exact comparator $L_{-i}$)", "-"),
                                 (gap, "gap", "true NashGap", "-")):
        ax.plot(t, vals, color=C[key], label=label, ls=ls, zorder=3,
                lw=1.5 if key == "gap" else 1.1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("NashGap or certificate")
    ax.legend(loc="upper right", handlelength=1.6, labelspacing=0.3)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.97)
    fig.savefig(os.path.join(OUT_DIR, "fig_coalition.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_coalition.png"))
    plt.close(fig)
    g = np.maximum(gap, 1e-300)
    share = (md - c1) / np.maximum(md - gap, 1e-300)
    print(f"final: gap {gap[-1]:.3e}  MD {md[-1]:.3e} ({md[-1]/g[-1]:.1f}x)  "
          f"C0 {c0[-1]:.3e} ({c0[-1]/g[-1]:.2f}x)  C1 {c1[-1]:.3e} ({c1[-1]/g[-1]:.2f}x)")
    print(f"share (MD-C1)/(MD-gap): final {share[-1]:.3f} median {np.median(share):.3f} "
          f"range {share.min():.3f}--{share.max():.3f}")
    print(f"tightness ranges: MD {(md/g).min():.1f}--{(md/g).max():.1f}x; "
          f"C0 {(c0/g).min():.2f}--{(c0/g).max():.2f}x; C1 {(c1/g).min():.2f}--{(c1/g).max():.2f}x")
    print(f"sound: C1 {bool(np.all(c1 >= gap))}, C0 {bool(np.all(c0 >= gap))}")


if __name__ == "__main__":
    main()

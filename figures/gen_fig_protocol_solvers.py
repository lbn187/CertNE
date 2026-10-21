#!/usr/bin/env python3
"""Vector schematic of shared checkpoints and two distinct selection rules.

All nodes, registers and strategy glyphs are illustrative, not numerical data.
The panel-alignment audit is optional: it runs when the ``audit_panel_alignment``
helper is on PYTHONPATH and is skipped otherwise.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

OUT = Path(__file__).resolve().parent / "fig_protocol"
W, H = 396, 180
INK, MUTED, LIGHT = "#243747", "#637583", "#DAE2E7"
BLUE, TEAL, AMBER = "#446C95", "#087F83", "#AF6C30"
PALE_TEAL, PALE_AMBER = "#F0F8F8", "#FBF6EF"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.4, "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
    "axes.spines.top": False, "axes.spines.right": False,
})


def text(ax, x, y, s, size=7.4, color=INK, ha="center", weight="normal"):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
                   fontweight=weight, zorder=10)


def line(ax, xs, ys, color=LIGHT, lw=0.8, ls="-"):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=2)


def arrow(ax, start, end, color=INK, lw=0.85, ls="-", rad=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
        mutation_scale=5.2, color=color, lw=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=3))


def rounded(ax, x, y, w, h, fc, ec="none", r=3, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc,
        edgecolor=ec, lw=lw, zorder=0))


def dot(ax, x, y, color=BLUE, r=1.7, fill=True):
    ax.add_patch(Circle((x, y), r, facecolor=color if fill else "white",
                        edgecolor=color, linewidth=0.65, zorder=5))


def tree(ax, cx, cy, color, signed=False, scale=1):
    """Backward arrows point from leaf values toward the root."""
    pts = [(0, 9), (-7, 0), (7, 0), (-11, -9), (-3, -9), (3, -9), (11, -9)]
    pts = [(cx + x * scale, cy + y * scale) for x, y in pts]
    for child, parent in [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2)]:
        a, b = pts[child], pts[parent]
        line(ax, [a[0], b[0]], [a[1], b[1]], color=color, lw=0.65)
    for i, (x, y) in enumerate(pts):
        dot(ax, x, y, color, r=1.35 * scale, fill=i < 3)
    for child, parent in [(4, 1), (1, 0)]:
        a, b = pts[child], pts[parent]
        start = (a[0] * 0.8 + b[0] * 0.2, a[1] * 0.8 + b[1] * 0.2)
        end = (a[0] * 0.2 + b[0] * 0.8, a[1] * 0.2 + b[1] * 0.8)
        arrow(ax, start, end, color, lw=1.0, ls="--" if signed else "-")
    if signed:
        for i, sign in [(3, "+"), (6, "−")]:
            x, y = pts[i]
            text(ax, x, y - 7, sign, size=6.2, color=color)


def strategy(ax, cx, cy, color):
    """Categorical policy icon; bar heights have no quantitative interpretation."""
    for x, height in [(-5, 6), (0, 13), (5, 9)]:
        rounded(ax, cx + x - 1.5, cy - 7, 3, height, color, r=0.8)
    line(ax, [cx - 8, cx + 8], [cy - 9, cy - 9], color=color, lw=0.6)


def shield(ax, cx, cy, color):
    vertices = [(cx - 8, cy + 7), (cx + 8, cy + 7), (cx + 7, cy - 3),
                (cx, cy - 9), (cx - 7, cy - 3)]
    ax.add_patch(Polygon(vertices, closed=True, facecolor="white",
                         edgecolor=color, linewidth=0.95, zorder=4))
    line(ax, [cx - 4, cx - 1, cx + 4], [cy, cy - 3, cy + 3], color, 1.2)


def selector(ax, cx, cy, color):
    ax.add_patch(Polygon([(cx, cy + 9), (cx + 9, cy), (cx, cy - 9),
                          (cx - 9, cy)], closed=True,
                         facecolor="white", edgecolor=color, lw=0.9, zorder=4))
    dot(ax, cx, cy, color, r=2.2)
    ax.add_patch(Circle((cx, cy), 4.5, fill=False, ec=color, lw=0.7, zorder=5))


def register(ax, cx, cy, symbol, kind):
    rounded(ax, cx - 11, cy - 10, 22, 20, "white", LIGHT, r=2)
    text(ax, cx, cy + 1, symbol, size=9, color=TEAL)
    if kind == "signed":
        line(ax, [cx - 7, cx - 2], [cy - 6, cy - 6], TEAL, 1.3)
        line(ax, [cx + 2, cx + 7], [cy - 6, cy - 6], AMBER, 1.3)
    elif kind == "value":
        dot(ax, cx - 5, cy - 6, TEAL, r=1)
        dot(ax, cx + 5, cy - 6, TEAL, r=1, fill=False)
    else:
        for x in [-6, -2, 2, 6]:
            line(ax, [cx + x, cx + x], [cy - 8, cy - 5], TEAL, 0.7)


def make_figure():
    fig = plt.figure(figsize=(W / 72, H / 72))
    background = fig.add_axes([0, 0, 1, 1], label="connectors")
    background.set(xlim=(0, W), ylim=(0, H)); background.axis("off")
    # Two equal comparison axes retain exact rendered alignment in points.
    axes = []
    for label, bottom, face in [("a", 103, PALE_AMBER), ("b", 25, PALE_TEAL)]:
        ax = fig.add_axes([100 / W, bottom / H, 292 / W, 74 / H], label=label)
        ax.set(xlim=(100, 392), ylim=(bottom, bottom + 74)); ax.axis("off")
        rounded(ax, 100, bottom, 292, 74, face, r=4)
        text(ax, 106, bottom + 65, label, 8.5, weight="bold")
        axes.append(ax)
    top, lower = axes
    a = background
    text(a, 43, 167, "Training-free", 8, weight="bold")
    text(a, 43, 156, "portfolio", 8, weight="bold")
    for y, label in [(134, r"$k_1$"), (113, r"$k_2$"), (71, r"$k_K$")]:
        text(a, 9, y, label, 7.4, MUTED)
        line(a, [24, 76], [y, y], LIGHT, 1)
        for x in [26, 38, 50, 62, 74]:
            dot(a, x, y, BLUE, r=2.1)
    text(a, 49, 93, "⋮", 12, MUTED)
    text(a, 45, 49, r"$K$ solvers", 7.4, MUTED)
    text(a, 45, 38, r"$m$ checkpoints", 7.4, MUTED)
    arrow(a, (25, 61), (76, 61), MUTED, 0.65)
    text(a, 48, 17, "Same snapshots", 6.5, MUTED)
    line(a, [80, 88, 88], [113, 113, 139], BLUE)
    arrow(a, (88, 139), (99, 139), AMBER)
    line(a, [88, 88], [113, 65], BLUE)
    arrow(a, (88, 65), (99, 65), TEAL)
    dot(a, 88, 113, BLUE, 1.6)

    text(top, 117, 168, "Exact evaluation", 8, AMBER, ha="left", weight="bold")
    text(top, 386, 168, r"$n$ BRs / checkpoint", 7, AMBER, ha="right")
    for x in [121, 154, 187]:
        tree(top, x, 139, AMBER)
        dot(top, x, 119, AMBER, r=2)
    line(top, [115, 204], [119, 119], AMBER, 0.6)
    text(top, 208, 137, "…", 9, AMBER)
    arrow(top, (220, 139), (276, 139), AMBER)
    selector(top, 287, 139, AMBER)
    text(top, 286, 119, r"$\arg\min\ g$", 7.5, AMBER)
    text(top, 164, 110, r"$Kmn$ BR passes", 7.4, AMBER)
    arrow(top, (298, 139), (312, 139), AMBER)
    strategy(top, 325, 141, AMBER)
    text(top, 361, 143, r"$\hat\sigma,\ \hat g$", 9, AMBER)
    text(top, 349, 119, r"$\hat g=\mathrm{NashGap}(\hat\sigma)$", 7.4, AMBER)

    text(lower, 117, 90, "Online certification", 8, TEAL, ha="left", weight="bold")
    text(lower, 386, 90, "0 BRs for selection", 7, TEAL, ha="right")
    for x, s, kind in [(118, r"$\widehat R$", "signed"), (146, r"$\Delta$", "value"),
                        (174, r"$w$", "weight")]:
        register(lower, x, 65, s, kind)
        line(lower, [x, x], [55, 46], TEAL, 0.65)
    line(lower, [118, 195], [46, 46], TEAL, 0.65)
    text(lower, 146, 34, "Accumulators", 6.4, MUTED)
    line(lower, [195, 195, 204], [46, 73, 73], TEAL, 0.7)
    arrow(lower, (195, 46), (208, 46), TEAL, 0.7)
    text(lower, 218, 74, r"$\|w\|_1$", 8.4, TEAL)
    text(lower, 249, 74, r"$C^0$", 8.4, TEAL)
    tree(lower, 219, 47, TEAL, signed=True, scale=0.64)
    text(lower, 249, 43, r"$C^1$", 8.4, TEAL)
    arrow(lower, (230, 73), (237, 73), TEAL, 0.7)
    arrow(lower, (230, 46), (237, 46), TEAL, 0.7)
    line(lower, [259, 268, 268], [74, 74, 59], TEAL, 0.7)
    line(lower, [259, 268, 268], [43, 43, 59], TEAL, 0.7)
    arrow(lower, (268, 59), (276, 59), TEAL, 0.7)
    selector(lower, 287, 59, TEAL)
    text(lower, 287, 39, r"$\arg\min\ C$", 7.5, TEAL)
    arrow(lower, (298, 59), (312, 59), TEAL)
    strategy(lower, 324, 65, TEAL)
    shield(lower, 349, 66, TEAL)
    text(lower, 375, 66, r"$\hat\sigma,\ \hat C$", 8.3, TEAL)
    text(lower, 350, 47, r"$\mathrm{NashGap}(\hat\sigma)\leq\hat C$", 7.4, TEAL)
    # Verification is a separate optional step, never fed back into selection.
    arrow(lower, (328, 39), (338, 31), MUTED, 0.7, ls="--")
    text(lower, 366, 31, "Verify once", 6.5, MUTED)

    # Shared pictographic key. All icons and timelines are schematic.
    dot(a, 111, 11, BLUE, 1.7)
    text(a, 118, 11, "Checkpoint", 6.2, MUTED, ha="left")
    arrow(a, (172, 11), (184, 11), AMBER, 0.9)
    text(a, 191, 11, "BR pass", 6.2, MUTED, ha="left")
    arrow(a, (234, 11), (246, 11), TEAL, 0.9, ls="--")
    text(a, 253, 11, "Signed pass", 6.2, MUTED, ha="left")
    text(a, 389, 11, "Schematic", 6.2, MUTED, ha="right")
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=["a", "b"],
            column_groups=[["a", "b"]], tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, strict=True,
            json_out=str(OUT) + ".alignment.json", overlay_svg=str(OUT) + ".alignment.svg")
    fig.savefig(str(OUT) + ".pdf", facecolor="white")
    fig.savefig(str(OUT) + ".svg", facecolor="white")
    fig.savefig(str(OUT) + ".png", dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()

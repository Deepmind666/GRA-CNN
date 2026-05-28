"""
GRA-CNN Pipeline Figure - Improved version with correct terminology.
Matches the final IVC manuscript terminology:
  - Structural Anchor: Independence · Taylor · Fisher
  - GRA Semantic Branch: Class-conditional Redundancy
  - Quality-Gated Boundary Refinement (with diamond gate)
  - Dependency-safe Structural Pruning
  - Fine-tune & Evaluate
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np
from pathlib import Path

# ── Color palette (per spec) ──
TXT   = "#1F2937"
BLUE  = "#1F4E79"
GRAY  = "#F3F4F6"
GREEN = "#2E7D32"
ICON  = "#4B5563"
WHITE = "#FFFFFF"
GOLD  = "#B8860B"   # quality gate accent

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "text.color": TXT,
})

fig = plt.figure(figsize=(16, 5.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16); ax.set_ylim(0, 5.5)
ax.axis("off"); fig.patch.set_facecolor(WHITE)

LW_B = 1.3; LW_A = 1.4; LW_I = 1.6


def box(x, y, w, h, ec=TXT, fc=GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=LW_B, zorder=2))


def arrow(x0, y0, x1, y1, c=TXT):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 color=c, lw=LW_A, mutation_scale=14, zorder=3))


def elbow(x0, y0, xm, x1, y1, c=TXT):
    ax.plot([x0, xm], [y0, y0], color=c, lw=LW_A, solid_capstyle="round", zorder=3)
    ax.plot([xm, xm], [y0, y1], color=c, lw=LW_A, solid_capstyle="round", zorder=3)
    arrow(xm, y1, x1, y1, c=c)


def title(x, y, txt, fs=11, fw="bold", c=TXT):
    ax.text(x, y, txt, ha="center", va="top", fontsize=fs, fontweight=fw, color=c)


def sub(x, y, txt, fs=9.5, c="#6B7280"):
    ax.text(x, y, txt, ha="center", va="top", fontsize=fs, color=c, style="italic")


# ── Mini icon: neural network ──
def icon_nn(cx, cy, s=0.4):
    layers = [3, 4, 4, 2]
    xs = np.linspace(cx - s, cx + s, 4)
    for i, (lx, n) in enumerate(zip(xs, layers)):
        ys = np.linspace(cy - s * 0.8, cy + s * 0.8, n)
        for yy in ys:
            ax.add_patch(plt.Circle((lx, yy), 0.06, fc="none", ec=ICON, lw=LW_I, zorder=4))
        if i < 3:
            ys2 = np.linspace(cy - s * 0.8, cy + s * 0.8, layers[i + 1])
            for y0 in ys:
                for y1 in ys2:
                    ax.plot([lx + 0.06, xs[i + 1] - 0.06], [y0, y1],
                            color=ICON, alpha=0.2, lw=0.5, zorder=3)


# ── Mini icon: bar chart (for structural anchor) ──
def icon_bars(cx, cy, s=0.35):
    vals = [0.85, 0.65, 0.50, 0.35, 0.20]
    xs = np.linspace(cx - s * 0.7, cx + s * 0.7, 5)
    colors_bar = [ICON, ICON, ICON, ICON, ICON]
    for x, v, cb in zip(xs, vals, colors_bar):
        h = v * s * 1.5
        ax.add_patch(plt.Rectangle((x - 0.04, cy - s * 0.5), 0.08, h,
                     fc="none", ec=cb, lw=LW_I, zorder=4))


# ── Mini icon: GRA trend curves (two class-conditional curves) ──
def icon_gra_trends(cx, cy, s=0.4):
    t = np.linspace(0, 1, 20)
    np.random.seed(42)
    # Class-1 trend
    a = cy - s * 0.3 + s * 0.7 * t + 0.02 * np.random.randn(20)
    # Class-2 trend (different pattern)
    b = cy + s * 0.1 - s * 0.3 * np.sin(t * np.pi) + 0.02 * np.random.randn(20)
    xs = np.linspace(cx - s, cx + s, 20)
    ax.plot(xs, a, color=BLUE, lw=LW_I, zorder=4)
    ax.plot(xs, b, color=BLUE, lw=LW_I, ls="--", zorder=4, alpha=0.7)
    # Small class labels
    ax.text(cx + s + 0.05, a[-1], r"$k_1$", fontsize=7, color=BLUE, va="center")
    ax.text(cx + s + 0.05, b[-1], r"$k_2$", fontsize=7, color=BLUE, va="center")


# ── Mini icon: diamond gate ──
def icon_diamond(cx, cy, s=0.25):
    pts = [(cx, cy + s), (cx + s, cy), (cx, cy - s), (cx - s, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc="#FFF8E1", ec=GOLD, lw=LW_I, zorder=4))
    ax.text(cx, cy, r"$Q{\geq}\tau$?", fontsize=8, ha="center", va="center",
            color=GOLD, fontweight="bold", zorder=5)


# ── Mini icon: boundary swap arrows ──
def icon_swap(cx, cy, s=0.3):
    # Two small opposing arrows representing swap
    ax.annotate("", xy=(cx + s * 0.6, cy + s * 0.3), xytext=(cx - s * 0.6, cy + s * 0.3),
                arrowprops=dict(arrowstyle="->", color=ICON, lw=1.2), zorder=4)
    ax.annotate("", xy=(cx - s * 0.6, cy - s * 0.3), xytext=(cx + s * 0.6, cy - s * 0.3),
                arrowprops=dict(arrowstyle="->", color=ICON, lw=1.2), zorder=4)
    ax.text(cx, cy + s * 0.55, "keep", fontsize=7, ha="center", va="bottom", color=ICON)
    ax.text(cx, cy - s * 0.55, "prune", fontsize=7, ha="center", va="top", color=ICON)


# ── Mini icon: scissors for pruning ──
def icon_scissors(cx, cy, s=0.3):
    ax.plot([cx - s, cx + s], [cy - s * 0.5, cy + s * 0.5], color=ICON, lw=LW_I, zorder=4)
    ax.plot([cx - s, cx + s], [cy + s * 0.5, cy - s * 0.5], color=ICON, lw=LW_I, zorder=4)
    ax.add_patch(plt.Circle((cx - s, cy - s * 0.5), 0.08, fc="none", ec=ICON, lw=LW_I, zorder=4))
    ax.add_patch(plt.Circle((cx - s, cy + s * 0.5), 0.08, fc="none", ec=ICON, lw=LW_I, zorder=4))


# ── Mini icon: training curve ──
def icon_chart(cx, cy, s=0.35):
    t = np.linspace(0, 1, 15)
    y = cy - s * 0.3 + s * 0.7 * (1 - np.exp(-t * 4))
    xs = np.linspace(cx - s, cx + s, 15)
    ax.plot(xs, y, color=ICON, lw=LW_I, zorder=4)
    ax.plot([cx - s, cx + s], [cy + s * 0.4, cy + s * 0.4], color=ICON, lw=0.7, ls=":", zorder=4)


# ══════════════════════════════════════════════
# Layout: 5 main columns
# ══════════════════════════════════════════════

mid_y = 2.75  # vertical center of the canvas

# ── Col 1: Input ──
bx1, bw1, bh1 = 0.3, 2.0, 2.2
by1 = mid_y - bh1 / 2
box(bx1, by1, bw1, bh1)
icon_nn(bx1 + bw1 / 2, by1 + bh1 * 0.38)
title(bx1 + bw1 / 2, by1 + bh1 - 0.12, "Pre-trained CNN")
sub(bx1 + bw1 / 2, by1 + bh1 - 0.42, "+ Calibration Data")

# ── Col 2 top: Structural Anchor ──
bx2, bw2, bh2 = 3.0, 2.6, 1.7
by_top = mid_y + 0.25
box(bx2, by_top, bw2, bh2)
icon_bars(bx2 + bw2 / 2, by_top + bh2 * 0.58)
title(bx2 + bw2 / 2, by_top + bh2 - 0.12, "Structural Anchor")
sub(bx2 + bw2 / 2, by_top + 0.22, r"Independence $\cdot$ Taylor $\cdot$ Fisher")

# ── Col 2 bottom: GRA Semantic Branch ──
by_bot = mid_y - 0.25 - bh2
box(bx2, by_bot, bw2, bh2, ec=BLUE)
icon_gra_trends(bx2 + bw2 / 2, by_bot + bh2 * 0.55)
title(bx2 + bw2 / 2, by_bot + bh2 - 0.12, "GRA Semantic Branch", c=BLUE)
sub(bx2 + bw2 / 2, by_bot + 0.22, "Class-conditional Redundancy", c=BLUE)

# ── Col 3: Quality-Gated Boundary Refinement ──
bx3, bw3, bh3 = 6.4, 2.4, 3.2
by3 = mid_y - bh3 / 2
box(bx3, by3, bw3, bh3, ec=GOLD, fc="#FFFDF5")
title(bx3 + bw3 / 2, by3 + bh3 - 0.12, "Quality-Gated\nBoundary Refinement")
# Diamond quality gate
icon_diamond(bx3 + bw3 / 2, by3 + bh3 * 0.52)
# Swap icon below gate
icon_swap(bx3 + bw3 / 2, by3 + bh3 * 0.18)

# ── Col 4: Dependency-safe Pruning ──
bx4, bw4, bh4 = 9.6, 2.0, 2.0
by4 = mid_y - bh4 / 2
box(bx4, by4, bw4, bh4)
icon_scissors(bx4 + bw4 / 2, by4 + bh4 * 0.58)
title(bx4 + bw4 / 2, by4 + bh4 - 0.15, "Dependency-safe\nStructural Pruning")

# ── Col 5: Fine-tune & Evaluate ──
bx5, bw5, bh5 = 12.4, 3.2, 2.0
by5 = mid_y - bh5 / 2
box(bx5, by5, bw5, bh5, ec=GREEN, fc="#E8F5E9")
icon_chart(bx5 + 0.55, by5 + bh5 * 0.6)
title(bx5 + bw5 / 2 + 0.2, by5 + bh5 - 0.12, "Fine-tune & Evaluate", c=GREEN)
# Metrics listed
ax.text(bx5 + bw5 / 2, by5 + 0.55, "Top-1 Acc  |  CR / PR  |  Paired $\\Delta$",
        ha="center", va="center", fontsize=9.5, color=GREEN)
ax.text(bx5 + bw5 / 2, by5 + 0.22, "iso-FLOPs  |  Bootstrap CI",
        ha="center", va="center", fontsize=9, color="#6B7280", style="italic")

# ══════════════════════════════════════════════
# Arrows
# ══════════════════════════════════════════════
mid_top = by_top + bh2 / 2
mid_bot = by_bot + bh2 / 2
mid3 = by3 + bh3 / 2

# Input -> dual branches
elbow(bx1 + bw1, mid_y, bx1 + bw1 + 0.25, bx2, mid_top, c=TXT)
elbow(bx1 + bw1, mid_y, bx1 + bw1 + 0.25, bx2, mid_bot, c=BLUE)

# Dual branches -> Quality-Gated Refinement
elbow(bx2 + bw2, mid_top, bx2 + bw2 + 0.3, bx3, mid3 + 0.5, c=TXT)
elbow(bx2 + bw2, mid_bot, bx2 + bw2 + 0.3, bx3, mid3 - 0.5, c=BLUE)

# Quality-Gated -> Pruning
arrow(bx3 + bw3, mid3, bx4, mid_y)

# Pruning -> Fine-tune
arrow(bx4 + bw4, mid_y, bx5, mid_y)

# ══════════════════════════════════════════════
# Stage labels (top)
# ══════════════════════════════════════════════
stage_y = 5.25
stages = [
    (bx1 + bw1 / 2, "Stage 0"),
    (bx2 + bw2 / 2, "Stage 1"),
    (bx3 + bw3 / 2, "Stage 2"),
    (bx4 + bw4 / 2, "Stage 3"),
    (bx5 + bw5 / 2, "Stage 4"),
]
for sx, sl in stages:
    ax.text(sx, stage_y, sl, ha="center", va="center", fontsize=9,
            color="#9CA3AF", fontweight="bold")

# ══════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════
lx = 4.0
ly = 0.15
ax.plot([lx, lx + 0.5], [ly, ly], color=BLUE, lw=LW_A, solid_capstyle="round")
arrow(lx + 0.4, ly, lx + 0.6, ly, c=BLUE)
ax.text(lx + 0.7, ly, "Semantic path (GRA)", fontsize=9, color=BLUE,
        fontweight="bold", va="center")

lx2 = lx + 3.8
ax.plot([lx2, lx2 + 0.5], [ly, ly], color=TXT, lw=LW_A, solid_capstyle="round")
arrow(lx2 + 0.4, ly, lx2 + 0.6, ly, c=TXT)
ax.text(lx2 + 0.7, ly, "Structural path", fontsize=9, color=TXT, va="center")

lx3 = lx2 + 3.0
pts_leg = [(lx3, ly + 0.12), (lx3 + 0.12, ly), (lx3, ly - 0.12), (lx3 - 0.12, ly)]
ax.add_patch(Polygon(pts_leg, closed=True, fc="#FFF8E1", ec=GOLD, lw=1.0, zorder=4))
ax.text(lx3 + 0.22, ly, "Quality gate", fontsize=9, color=GOLD,
        fontweight="bold", va="center")

# ══════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════
out_dir = Path("paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_pipeline_poster.pdf",
            bbox_inches="tight", pad_inches=0.05, dpi=300)
fig.savefig(out_dir / "fig_pipeline_poster.png",
            bbox_inches="tight", pad_inches=0.05, dpi=150)
fig.savefig(out_dir / "fig_pipeline_poster_600dpi.png",
            bbox_inches="tight", pad_inches=0.05, dpi=600)
print("Saved: fig_pipeline_poster.pdf + png")
plt.close()

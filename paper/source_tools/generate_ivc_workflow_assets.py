from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
GRAY = "#3F3F46"
BLUE = "#7BA7F7"
BLUE_DARK = "#4F77D9"
ORANGE = "#E3B16A"
ORANGE_DARK = "#B7791F"
GREEN = "#8BC7A0"
GREEN_DARK = "#2E8B57"
SAGE = "#B9CCC2"
RED = "#D97757"


def add_box(ax, x, y, w, h, title, body, facecolor, title_size=12.5, body_size=10.1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.3,
        edgecolor=GRAY,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + h - 0.06,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=GRAY,
        va="top",
        linespacing=1.0,
    )
    ax.text(x + 0.02, y + h - 0.12, body, fontsize=body_size, color=GRAY, va="top", linespacing=1.0)


def add_arrow(ax, start, end):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color=GRAY)
    ax.add_patch(patch)


def save(fig, pdf_path: Path, png_path: Path) -> None:
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def add_icon_box(ax, x, y, w, h, title, body, facecolor, icon_fn):
    add_box(ax, x, y, w, h, title, body, facecolor, title_size=10.8, body_size=8.9)
    icon_fn(ax, x + 0.025, y + 0.07, w - 0.05, 0.16)


def draw_feature_icon(ax, x, y, w, h):
    cell = min(w * 0.14, h * 0.28)
    gap = cell * 0.25
    gx = x + 0.01
    gy = y + 0.02
    palette = ["#DCEBFF", "#C5DBFF", BLUE]
    for row in range(3):
        for col in range(3):
            ax.add_patch(
                Rectangle(
                    (gx + col * (cell + gap), gy + row * (cell + gap)),
                    cell,
                    cell,
                    linewidth=0.8,
                    edgecolor=BLUE_DARK,
                    facecolor=palette[(row + col) % len(palette)],
                )
            )
    xs = [
        x + w * 0.58,
        x + w * 0.68,
        x + w * 0.78,
        x + w * 0.88,
        x + w * 0.98,
    ]
    ys = [
        y + h * 0.25,
        y + h * 0.38,
        y + h * 0.33,
        y + h * 0.55,
        y + h * 0.48,
    ]
    ax.plot(xs, ys, color=BLUE_DARK, linewidth=2.0, solid_capstyle="round")
    for px, py in zip(xs, ys):
        ax.add_patch(Circle((px, py), h * 0.045, facecolor=BLUE, edgecolor=BLUE_DARK, linewidth=0.8))


def draw_anchor_icon(ax, x, y, w, h):
    bar_w = w * 0.12
    start_x = x + w * 0.10
    base_y = y + h * 0.12
    heights = [h * 0.33, h * 0.58, h * 0.46]
    for idx, bar_h in enumerate(heights):
        ax.add_patch(
            Rectangle(
                (start_x + idx * bar_w * 1.5, base_y),
                bar_w,
                bar_h,
                linewidth=0.8,
                edgecolor=ORANGE_DARK,
                facecolor=ORANGE,
            )
        )
    ax.plot(
        [x + w * 0.62, x + w * 0.62],
        [y + h * 0.16, y + h * 0.70],
        color=ORANGE_DARK,
        linewidth=1.2,
    )
    for px, py in [
        (x + w * 0.62, y + h * 0.62),
        (x + w * 0.74, y + h * 0.48),
        (x + w * 0.86, y + h * 0.31),
    ]:
        ax.add_patch(Circle((px, py), h * 0.05, facecolor="#FFF2DD", edgecolor=ORANGE_DARK, linewidth=0.9))
    ax.plot(
        [x + w * 0.62, x + w * 0.74, x + w * 0.86],
        [y + h * 0.62, y + h * 0.48, y + h * 0.31],
        color=ORANGE_DARK,
        linewidth=1.6,
    )


def draw_boundary_icon(ax, x, y, w, h):
    channel_w = w * 0.11
    gap = w * 0.055
    base_x = x + w * 0.08
    base_y = y + h * 0.12
    heights = [h * 0.34, h * 0.52, h * 0.46, h * 0.55]
    fills = ["#D7F0DF", GREEN, GREEN, "#D7F0DF"]
    for idx, bar_h in enumerate(heights):
        ax.add_patch(
            Rectangle(
                (base_x + idx * (channel_w + gap), base_y),
                channel_w,
                bar_h,
                linewidth=0.9,
                edgecolor=GREEN_DARK,
                facecolor=fills[idx],
            )
        )
    boundary_x = base_x + 2 * (channel_w + gap) - gap * 0.5
    ax.plot(
        [boundary_x, boundary_x],
        [y + h * 0.10, y + h * 0.72],
        color=GREEN_DARK,
        linewidth=1.2,
        linestyle="--",
    )
    gate = Circle((x + w * 0.77, y + h * 0.60), h * 0.09, facecolor="#ECF9F0", edgecolor=GREEN_DARK, linewidth=1.1)
    ax.add_patch(gate)
    ax.plot(
        [x + w * 0.74, x + w * 0.77, x + w * 0.81],
        [y + h * 0.60, y + h * 0.56, y + h * 0.65],
        color=GREEN_DARK,
        linewidth=1.6,
        solid_capstyle="round",
    )
    ax.add_patch(
        FancyArrowPatch(
            (base_x + channel_w * 1.45 + gap, y + h * 0.30),
            (base_x + channel_w * 2.5 + gap * 1.6, y + h * 0.42),
            arrowstyle="<->",
            mutation_scale=12,
            linewidth=1.3,
            color=GREEN_DARK,
            connectionstyle="arc3,rad=0.25",
        )
    )


def draw_pruned_model_icon(ax, x, y, w, h):
    cols = [x + w * 0.18, x + w * 0.50, x + w * 0.82]
    rows = [y + h * 0.22, y + h * 0.42, y + h * 0.62]
    radius = h * 0.045
    for left in cols[:2]:
        for row in rows:
            ax.add_patch(Circle((left, row), radius, facecolor="#F7FBF8", edgecolor=GREEN_DARK, linewidth=0.9))
    for row in [rows[0], rows[2]]:
        ax.add_patch(Circle((cols[2], row), radius, facecolor="#F7FBF8", edgecolor=GREEN_DARK, linewidth=0.9))
    pruned = Circle((cols[2], rows[1]), radius, facecolor="#F6E3DD", edgecolor=RED, linewidth=1.0)
    ax.add_patch(pruned)
    ax.plot(
        [cols[2] - radius * 0.7, cols[2] + radius * 0.7],
        [rows[1] - radius * 0.7, rows[1] + radius * 0.7],
        color=RED,
        linewidth=1.2,
    )
    ax.plot(
        [cols[2] - radius * 0.7, cols[2] + radius * 0.7],
        [rows[1] + radius * 0.7, rows[1] - radius * 0.7],
        color=RED,
        linewidth=1.2,
    )
    for left, right in [(cols[0], cols[1]), (cols[1], cols[2])]:
        for row_l, row_r in zip(rows, rows):
            if right == cols[2] and row_r == rows[1]:
                continue
            ax.plot([left + radius, right - radius], [row_l, row_r], color=SAGE, linewidth=1.0)


def generate_workflow() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 0.03, 0.28, 0.20, 0.42, "Calibration Features", "Pre-trained CNN\n+\nclass-balanced split", "#EEF5FF")
    add_box(ax, 0.29, 0.18, 0.23, 0.60, "Structural Anchor", "Channel independence\nTaylor sensitivity\nFisher information", "#FFF3E8")
    add_box(ax, 0.57, 0.18, 0.23, 0.60, "Semantic Boundary Refinement", "Class-aware GRA redundancy\nquality gate\nboundary swap correction", "#EEF8F1")
    add_box(ax, 0.84, 0.28, 0.13, 0.42, "Pruned CNN", "Dependency-safe pruning\n+\n30-epoch recovery", "#F4F4F5")

    add_arrow(ax, (0.23, 0.49), (0.29, 0.49))
    add_arrow(ax, (0.52, 0.49), (0.57, 0.49))
    add_arrow(ax, (0.80, 0.49), (0.84, 0.49))

    ax.text(0.69, 0.08, "Semantic scoring is applied only when the boundary signal is reliable.", fontsize=11.0, color="#0B5FA5", ha="center", fontweight="bold")
    save(fig, FIG_DIR / "fig_ivc_workflow.pdf", FIG_DIR / "fig_ivc_workflow.png")


def generate_graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(13.28, 5.31))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    y = 0.24
    h = 0.52
    blocks = [
        (0.05, 0.19, "Calibration\nFeatures", "Class-balanced feature responses", "#EEF5FF", draw_feature_icon),
        (0.27, 0.21, "Structural\nAnchor", "Independence + Taylor + Fisher", "#FFF4E8", draw_anchor_icon),
        (0.51, 0.22, "Quality-Gated\nBoundary Refinement", "Class-aware GRA on uncertain channels", "#EEF8F1", draw_boundary_icon),
        (0.76, 0.19, "Robust Pruned\nModel", "Stressed residual CNN settings", "#F2F7F4", draw_pruned_model_icon),
    ]

    for x, w, title, body, facecolor, icon_fn in blocks:
        add_icon_box(ax, x, y, w, h, title, body, facecolor, icon_fn)

    centers = [(x + w, y + h * 0.52) for x, w, *_ in blocks]
    starts = [centers[0], centers[1], centers[2]]
    ends = [
        (blocks[1][0], y + h * 0.52),
        (blocks[2][0], y + h * 0.52),
        (blocks[3][0], y + h * 0.52),
    ]
    for start, end in zip(starts, ends):
        add_arrow(ax, start, end)

    ax.text(
        0.50,
        0.09,
        "Semantic-aware structured pruning for robust visual model compression",
        fontsize=11.4,
        fontweight="bold",
        color=GREEN_DARK,
        ha="center",
    )
    fig.savefig(ROOT / "graphical_abstract.pdf", facecolor="white")
    fig.savefig(ROOT / "graphical_abstract.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_workflow()
    generate_graphical_abstract()

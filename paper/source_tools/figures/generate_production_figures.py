from pathlib import Path
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parents[1]
RESULT_DIR = ROOT / "experiments" / "chip_results"

METHODS = ["GRA-CNN (Ours)", "CHIP", "L1", "Taylor"]
METHOD_KEY = {
    "GRA-CNN (Ours)": "GRA-CHIP-v3",
    "CHIP": "CHIP",
    "L1": "L1",
    "Taylor": "Taylor",
}
COLORS = {
    "GRA-CNN (Ours)": "#0B5FA5",
    "CHIP": "#F28E2B",
    "L1": "#59A14F",
    "Taylor": "#B07AA1",
}
CELLS = [
    ("resnet56", 0.7, "ResNet-56, p=0.7"),
    ("resnet56", 0.9, "ResNet-56, p=0.9"),
    ("vgg16", 0.7, "VGG-16, p=0.7"),
    ("vgg16", 0.9, "VGG-16, p=0.9"),
]


def _load_rows():
    rows = []
    for p in RESULT_DIR.glob("*.json"):
        if p.name.endswith("_meta.json") or p.name.startswith("eb_"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("dataset") != "cifar100":
            continue
        if data.get("architecture") not in ("resnet56", "vgg16"):
            continue
        if data.get("method") not in METHOD_KEY.values():
            continue
        if float(data.get("target_ratio", -1)) not in (0.7, 0.9):
            continue
        rows.append(data)
    return rows


def _values(rows, arch, ratio, method):
    key = METHOD_KEY[method]
    vals = [
        float(r["final_acc"])
        for r in rows
        if r.get("architecture") == arch
        and float(r.get("target_ratio")) == ratio
        and r.get("method") == key
    ]
    cr = [
        float(r["compression_ratio"])
        for r in rows
        if r.get("architecture") == arch
        and float(r.get("target_ratio")) == ratio
        and r.get("method") == key
    ]
    return vals, (statistics.mean(cr) if cr else np.nan)


def _paired_delta(rows, arch, ratio, method_a, method_b):
    key_a = METHOD_KEY[method_a]
    key_b = METHOD_KEY[method_b]
    by_seed = {}
    for r in rows:
        if r.get("architecture") != arch or float(r.get("target_ratio")) != ratio:
            continue
        seed = int(r.get("seed", -1))
        if seed not in by_seed:
            by_seed[seed] = {}
        by_seed[seed][r.get("method")] = float(r["final_acc"])
    deltas = []
    for _, item in sorted(by_seed.items()):
        if key_a in item and key_b in item:
            deltas.append(item[key_a] - item[key_b])
    return deltas


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.2,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _mean_ci95(vals):
    if not vals:
        return np.nan, np.nan
    mean = statistics.mean(vals)
    if len(vals) == 1:
        return mean, 0.0
    std = statistics.stdev(vals)
    return mean, 1.96 * std / np.sqrt(len(vals))


def generate_main_figure():
    rows = _load_rows()
    if not rows:
        raise RuntimeError("No valid result rows found.")

    _style()
    fig = plt.figure(figsize=(11.0, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.12], wspace=0.14)

    # (a) Forest-style paired delta plot: GRA-CNN - CHIP
    ax0 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(CELLS))[::-1]
    rng = np.random.default_rng(1234)

    for i, (arch, ratio, label) in enumerate(CELLS):
        deltas = _paired_delta(rows, arch, ratio, "GRA-CNN (Ours)", "CHIP")
        mean, ci = _mean_ci95(deltas)
        y = y_pos[i]

        # seed-level paired deltas
        jitter = rng.normal(0.0, 0.045, size=len(deltas))
        ax0.scatter(
            deltas,
            np.full(len(deltas), y) + jitter,
            s=24,
            color="#9ca3af",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )

        color = "#0B5FA5" if arch == "resnet56" else "#6b7280"
        ax0.errorbar(
            mean,
            y,
            xerr=ci,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=3.0,
            markersize=6.2,
            zorder=3,
        )

    ax0.axvline(0.0, color="#374151", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels([c[2] for c in CELLS])
    ax0.set_xlabel("Paired delta vs CHIP (Top-1, percentage points)")
    ax0.set_title("(a) Paired delta with 95% confidence intervals", loc="left", pad=6)
    ax0.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax0.set_xlim(-0.65, 0.95)

    # (b) Accuracy-compression Pareto with compact method curves
    ax1 = fig.add_subplot(gs[0, 1])
    marker_map = {"GRA-CNN (Ours)": "o", "CHIP": "s", "L1": "^", "Taylor": "D"}
    line_style = {"resnet56": "-", "vgg16": "--"}
    arch_label = {"resnet56": "ResNet-56", "vgg16": "VGG-16"}

    for method in METHODS:
        for arch in ("resnet56", "vgg16"):
            xs, ys, yerr = [], [], []
            for ratio in (0.7, 0.9):
                vals, cr = _values(rows, arch, ratio, method)
                if vals:
                    xs.append(cr)
                    ys.append(statistics.mean(vals))
                    yerr.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            if len(xs) == 2:
                ax1.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    color=COLORS[method],
                    linestyle=line_style[arch],
                    marker=marker_map[method],
                    linewidth=1.4,
                    markersize=5.2,
                    capsize=2.6,
                    alpha=0.95,
                )

    # concise legend (method colors + architecture line styles)
    from matplotlib.lines import Line2D
    method_handles = [
        Line2D([0], [0], color=COLORS[m], marker=marker_map[m], linestyle="-", linewidth=1.4, markersize=5.2, label=m)
        for m in METHODS
    ]
    arch_handles = [
        Line2D([0], [0], color="#111827", linestyle=line_style[a], linewidth=1.4, label=arch_label[a])
        for a in ("resnet56", "vgg16")
    ]
    leg1 = ax1.legend(handles=method_handles, loc="lower left", frameon=False, ncol=2, title="Methods")
    ax1.add_artist(leg1)
    ax1.legend(handles=arch_handles, loc="upper right", frameon=False, title="Line style")

    ax1.set_xscale("log")
    ax1.set_xlabel("Compression ratio (x)")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_title("(b) Accuracy-compression Pareto by method and architecture", loc="left", pad=6)
    ax1.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.35)

    fig.savefig(OUT_DIR / "fig_main_accuracy.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_main_accuracy.png", dpi=700, bbox_inches="tight")


if __name__ == "__main__":
    generate_main_figure()
    print("Saved: fig_main_accuracy.{pdf,png}")

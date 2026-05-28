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

METHOD_STYLE = {
    "GRA-CHIP-v3": {"label": "GRA-CNN (Ours)", "color": "#0B5FA5", "marker": "o", "linestyle": "-", "zorder": 7},
    "CHIP": {"label": "CHIP", "color": "#F28E2B", "marker": "s", "linestyle": "--", "zorder": 6},
    "L1": {"label": "L1-Norm", "color": "#2CA02C", "marker": "^", "linestyle": "-.", "zorder": 5},
    "Taylor": {"label": "Taylor", "color": "#9467BD", "marker": "D", "linestyle": ":", "zorder": 4},
    "FPGM": {"label": "FPGM", "color": "#8C564B", "marker": "v", "linestyle": "-", "zorder": 3},
    "HRank": {"label": "HRank", "color": "#17BECF", "marker": "P", "linestyle": "--", "zorder": 2},
}


def _load_rows():
    rows = []
    for p in RESULT_DIR.glob("*.json"):
        if p.name.endswith("_meta.json") or p.name.startswith("eb_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "final_acc" not in d:
            continue
        rows.append(d)
    return rows


def _rows(rows, arch, dataset, method):
    out = []
    for r in rows:
        if r.get("architecture") != arch:
            continue
        if r.get("dataset") != dataset:
            continue
        if r.get("method") != method:
            continue
        ratio = float(r.get("target_ratio", -1))
        out.append((ratio, int(r.get("seed", -1)), float(r["final_acc"])))
    return out


def _group_curve(rows, arch, dataset, method):
    vals = _rows(rows, arch, dataset, method)
    by_ratio = {}
    for ratio, _, acc in vals:
        if ratio not in by_ratio:
            by_ratio[ratio] = []
        by_ratio[ratio].append(acc)
    xs = sorted(by_ratio.keys())
    ys = [statistics.mean(by_ratio[x]) for x in xs]
    ystd = [statistics.stdev(by_ratio[x]) if len(by_ratio[x]) > 1 else 0.0 for x in xs]
    return xs, ys, ystd


def _paired_delta(rows, arch, dataset, ratio, method_a, method_b):
    by_seed = {}
    for r in rows:
        if r.get("architecture") != arch or r.get("dataset") != dataset:
            continue
        if float(r.get("target_ratio", -1)) != ratio:
            continue
        m = r.get("method")
        if m not in (method_a, method_b):
            continue
        s = int(r.get("seed", -1))
        if s not in by_seed:
            by_seed[s] = {}
        by_seed[s][m] = float(r["final_acc"])

    deltas = []
    for s in sorted(by_seed):
        item = by_seed[s]
        if method_a in item and method_b in item:
            deltas.append(item[method_a] - item[method_b])
    return deltas


def _mean_ci95(vals):
    if not vals:
        return np.nan, np.nan
    mean = statistics.mean(vals)
    if len(vals) == 1:
        return mean, 0.0
    std = statistics.stdev(vals)
    return mean, 1.96 * std / np.sqrt(len(vals))


def _set_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.8,
            "axes.linewidth": 1.0,
            "axes.facecolor": "#F9FAFB",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_multicell_curve(rows):
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 9.4), constrained_layout=False)
    panels = [
        ("resnet56", "cifar100", "(a) ResNet-56 on CIFAR-100"),
        ("vgg16", "cifar100", "(b) VGG-16 on CIFAR-100"),
    ]
    rng = np.random.default_rng(20260301)

    for ax, (arch, ds, title) in zip(np.atleast_1d(axes), panels):
        y_all = []
        ratios = sorted(
            set(
                float(r.get("target_ratio", -1))
                for r in rows
                if r.get("architecture") == arch and r.get("dataset") == ds
            )
        )

        for method, style in METHOD_STYLE.items():
            xs, ys, ystd = _group_curve(rows, arch, ds, method)
            if not xs:
                continue
            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            ystd_arr = np.array(ystd)
            y_all.extend(ys)

            ax.plot(
                xs_arr,
                ys_arr,
                linestyle=style["linestyle"],
                marker=style["marker"],
                color=style["color"],
                linewidth=2.4,
                markersize=7.3,
                label=style["label"],
                zorder=style["zorder"],
            )
            ax.fill_between(
                xs_arr,
                ys_arr - ystd_arr,
                ys_arr + ystd_arr,
                color=style["color"],
                alpha=0.15,
                linewidth=0.0,
                zorder=style["zorder"] - 1,
            )
            # seed-level jittered points for distribution visibility
            vals = _rows(rows, arch, ds, method)
            for x in xs_arr:
                accs = [acc for ratio, _, acc in vals if ratio == x]
                if not accs:
                    continue
                xj = np.full(len(accs), x) + rng.normal(0.0, 0.004, size=len(accs))
                yj = np.array(accs) + rng.normal(0.0, 0.02, size=len(accs))
                ax.scatter(
                    xj,
                    yj,
                    s=14,
                    color=style["color"],
                    alpha=0.55,
                    edgecolors="white",
                    linewidths=0.2,
                    zorder=style["zorder"] - 0.2,
                )
            for x, y in zip(xs_arr, ys_arr):
                ax.text(
                    x,
                    y + 0.16,
                    f"{y:.2f}",
                    color=style["color"],
                    fontsize=8.5,
                    ha="center",
                    va="bottom",
                )

        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("Pruning ratio")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.38)

        if ratios:
            ax.set_xticks(ratios)
            ax.set_xlim(min(ratios) - 0.045, max(ratios) + 0.045)

        if y_all:
            y_min = min(y_all)
            y_max = max(y_all)
            # VGG-16 contains a known extreme-compression outlier (L1 at p=0.9),
            # so we clamp the axis for readability and annotate clipped points.
            if arch == "vgg16" and y_min < 35.0:
                y_low, y_high = 40.0, y_max + 1.6
                ax.set_ylim(y_low, y_high)
                for method, style in METHOD_STYLE.items():
                    vals = _rows(rows, arch, ds, method)
                    for x in ratios:
                        accs = [acc for ratio, _, acc in vals if ratio == x]
                        if not accs:
                            continue
                        m = statistics.mean(accs)
                        if m < y_low:
                            ax.annotate(
                                "",
                                xy=(x, y_low + 0.25),
                                xytext=(x, y_low + 1.5),
                                arrowprops=dict(arrowstyle="-|>", color=style["color"], lw=1.5),
                            )
                            ax.text(
                                x,
                                y_low + 1.75,
                                f"{style['label']}={m:.2f}",
                                color=style["color"],
                                fontsize=8.1,
                                ha="center",
                                va="bottom",
                            )
            else:
                pad = max(0.9, (y_max - y_min) * 0.30)
                ax.set_ylim(y_min - pad, y_max + pad)

        # highlight high-compression area
        ax.axvspan(0.86, 0.94, color="#E6F0FF", alpha=0.55, zorder=0)
        # panel note: GRA-CNN vs CHIP delta by ratio
        d07 = _paired_delta(rows, arch, ds, 0.7, "GRA-CHIP-v3", "CHIP")
        d09 = _paired_delta(rows, arch, ds, 0.9, "GRA-CHIP-v3", "CHIP")
        if d07 and d09:
            t = f"$\\Delta_{{0.7}}$={statistics.mean(d07):+.2f} pp\n$\\Delta_{{0.9}}$={statistics.mean(d09):+.2f} pp"
            ax.text(
                0.03,
                0.96,
                t,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.0,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D1D5DB", alpha=0.9),
            )

    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=6,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.1,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.15, wspace=0.12)
    fig.savefig(OUT_DIR / "fig_multicell_curves.pdf", format="pdf")
    fig.savefig(OUT_DIR / "fig_multicell_curves.png", dpi=700)


def make_residual_main_curve(rows):
    """High-signal main figure: ResNet-56/CIFAR-100 across p=0.6/0.7/0.8/0.9."""
    _set_style()
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.8), constrained_layout=False)
    arch, ds = "resnet56", "cifar100"
    rng = np.random.default_rng(20260301)

    y_all = []
    ratios = sorted(
        set(
            float(r.get("target_ratio", -1))
            for r in rows
            if r.get("architecture") == arch and r.get("dataset") == ds
        )
    )

    for method, style in METHOD_STYLE.items():
        xs, ys, ystd = _group_curve(rows, arch, ds, method)
        if not xs:
            continue
        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        ystd_arr = np.array(ystd)
        y_all.extend(ys)

        ax.plot(
            xs_arr,
            ys_arr,
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.5,
            markersize=7.8,
            label=style["label"],
            zorder=style["zorder"],
        )
        ax.fill_between(
            xs_arr,
            ys_arr - ystd_arr,
            ys_arr + ystd_arr,
            color=style["color"],
            alpha=0.14,
            linewidth=0.0,
            zorder=style["zorder"] - 1,
        )
        vals = _rows(rows, arch, ds, method)
        for x in xs_arr:
            accs = [acc for ratio, _, acc in vals if ratio == x]
            if not accs:
                continue
            xj = np.full(len(accs), x) + rng.normal(0.0, 0.0035, size=len(accs))
            yj = np.array(accs) + rng.normal(0.0, 0.015, size=len(accs))
            ax.scatter(
                xj,
                yj,
                s=15,
                color=style["color"],
                alpha=0.55,
                edgecolors="white",
                linewidths=0.2,
                zorder=style["zorder"] - 0.2,
            )

    ax.set_title("ResNet-56 on CIFAR-100 (Primary Residual Matrix)", loc="left", pad=8)
    ax.set_xlabel("Pruning ratio")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.grid(True, linestyle="--", linewidth=0.72, alpha=0.38)
    if ratios:
        ax.set_xticks(ratios)
        ax.set_xlim(min(ratios) - 0.04, max(ratios) + 0.04)
    if y_all:
        y_min = min(y_all)
        y_max = max(y_all)
        pad = max(0.8, (y_max - y_min) * 0.22)
        ax.set_ylim(y_min - pad, y_max + pad)

    # Highlight the high-compression region where GRA-CNN shows strongest gains vs CHIP.
    ax.axvspan(0.68, 0.92, color="#E6F0FF", alpha=0.50, zorder=0)
    d07 = _paired_delta(rows, arch, ds, 0.7, "GRA-CHIP-v3", "CHIP")
    d08 = _paired_delta(rows, arch, ds, 0.8, "GRA-CHIP-v3", "CHIP")
    d09 = _paired_delta(rows, arch, ds, 0.9, "GRA-CHIP-v3", "CHIP")
    if d07 and d08 and d09:
        note = (
            f"$\\Delta_{{0.7}}$={statistics.mean(d07):+.2f} pp\n"
            f"$\\Delta_{{0.8}}$={statistics.mean(d08):+.2f} pp\n"
            f"$\\Delta_{{0.9}}$={statistics.mean(d09):+.2f} pp"
        )
        ax.text(
            0.03,
            0.96,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#D1D5DB", alpha=0.92),
        )

    ax.legend(
        ncol=3,
        loc="lower left",
        frameon=False,
        bbox_to_anchor=(0.0, -0.02),
        handlelength=2.0,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.12)
    fig.savefig(OUT_DIR / "fig_residual_main.pdf", format="pdf")
    fig.savefig(OUT_DIR / "fig_residual_main.png", dpi=700)


def make_primary_forest(rows):
    _set_style()
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 10.2), constrained_layout=False, sharey=True)
    cells = [
        ("resnet56", "cifar100", 0.7, "ResNet-56, p=0.7"),
        ("resnet56", "cifar100", 0.9, "ResNet-56, p=0.9"),
        ("vgg16", "cifar100", 0.7, "VGG-16, p=0.7"),
        ("vgg16", "cifar100", 0.9, "VGG-16, p=0.9"),
    ]
    y_pos = np.arange(len(cells))[::-1]
    rng = np.random.default_rng(20260228)

    def draw_panel(ax, baseline_method, title, xlim):
        for i, (arch, ds, ratio, _) in enumerate(cells):
            deltas = _paired_delta(rows, arch, ds, ratio, "GRA-CHIP-v3", baseline_method)
            mean, ci = _mean_ci95(deltas)
            y = y_pos[i]
            jitter = rng.normal(0.0, 0.05, size=len(deltas))

            ax.scatter(
                deltas,
                np.full(len(deltas), y) + jitter,
                s=28,
                color="#BFC7D5",
                edgecolors="white",
                linewidths=0.35,
                alpha=0.92,
                zorder=2,
            )

            sig = (not np.isnan(mean)) and (not np.isnan(ci)) and ((mean - ci > 0) or (mean + ci < 0))
            if sig:
                color = "#0B5FA5" if arch == "resnet56" else "#0F766E"
                face = color
            else:
                color = "#6b7280"
                face = "white"

            lo = mean - ci
            hi = mean + ci
            clip_lo = max(lo, xlim[0])
            clip_hi = min(hi, xlim[1])
            off_right = mean > xlim[1]
            off_left = mean < xlim[0]
            plot_mean = min(max(mean, xlim[0] + 0.02), xlim[1] - 0.02)

            ax.errorbar(
                plot_mean,
                y,
                xerr=[[max(0.0, plot_mean - clip_lo)], [max(0.0, clip_hi - plot_mean)]],
                fmt="o",
                color=color,
                ecolor=color,
                markerfacecolor=face,
                elinewidth=2.0,
                capsize=3.6,
                markersize=7.2,
                zorder=4,
            )
            if off_right or off_left:
                edge_x = xlim[1] - 0.02 if off_right else xlim[0] + 0.02
                direction = 0.18 if off_right else -0.18
                ax.annotate(
                    "",
                    xy=(edge_x, y),
                    xytext=(edge_x - direction, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                    zorder=5,
                )
                ax.text(
                    edge_x + (0.02 if off_right else -0.02),
                    y + 0.21,
                    f"{mean:+.2f}",
                    fontsize=8.6,
                    color=color,
                    ha="left" if off_right else "right",
                    va="center",
                )
            elif lo < xlim[0] or hi > xlim[1]:
                ax.annotate(
                    "",
                    xy=(clip_hi if hi > xlim[1] else clip_lo, y),
                    xytext=(plot_mean, y),
                    arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.2),
                    zorder=3,
                )

            if not np.isnan(mean):
                ax.text(
                    plot_mean + (0.055 if plot_mean >= 0 else -0.055),
                    y + 0.22,
                    f"{mean:+.2f}",
                    fontsize=8.8,
                    color=color,
                    ha="left" if mean >= 0 else "right",
                    va="center",
                )

        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.15, zorder=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([c[3] for c in cells])
        ax.set_xlabel("Paired delta (percentage points)")
        ax.set_title(title, loc="left", pad=8)
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_xlim(*xlim)

    draw_panel(axes[0], "CHIP", "(a) GRA-CNN vs CHIP", (-0.95, 1.25))
    draw_panel(axes[1], "L1", "(b) GRA-CNN vs L1-Norm", (-2.2, 3.0))

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="o", color="#0B5FA5", linestyle="", markersize=7.2, label="CI excludes 0"),
        Line2D([0], [0], marker="o", color="#6b7280", markerfacecolor="white", linestyle="", markersize=7.2, label="CI overlaps 0"),
        Line2D([0], [0], marker="o", color="#b8c0cc", linestyle="", markersize=6.0, label="Seed-level deltas"),
    ]
    fig.legend(
        legend_items,
        [i.get_label() for i in legend_items],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.92, bottom=0.08, hspace=0.36)
    fig.savefig(OUT_DIR / "fig_primary_forest.pdf", format="pdf")
    fig.savefig(OUT_DIR / "fig_primary_forest.png", dpi=700)


if __name__ == "__main__":
    rows = _load_rows()
    make_residual_main_curve(rows)
    make_multicell_curve(rows)
    make_primary_forest(rows)
    print("Saved: fig_residual_main, fig_multicell_curves, fig_primary_forest (pdf/png)")

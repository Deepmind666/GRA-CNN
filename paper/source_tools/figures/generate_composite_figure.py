"""Generate a 2x3 composite experimental results figure for the IVC submission."""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parents[1]
RESULT_DIR = ROOT / "experiments" / "chip_results"
UNIFIED_DIR = ROOT / "experiments" / "unified_results"

# Canonical method key -> all known aliases in JSON files
METHOD_ALIASES = {
    "GRA-CHIP-v3": ["GRA-CHIP-v3", "GRA-CNN", "GRA-CHIP-v3.1", "GRA-CHIP-v3.2", "GRA-v7"],
    "CHIP": ["CHIP"],
    "L1": ["L1"],
    "Taylor": ["Taylor"],
    "FPGM": ["FPGM"],
    "HRank": ["HRank"],
}

METHOD_STYLE = {
    "GRA-CHIP-v3": {"label": "GRA-CNN", "color": "#0B5FA5", "marker": "o", "ls": "-", "z": 7},
    "CHIP":        {"label": "CHIP",     "color": "#F28E2B", "marker": "s", "ls": "--", "z": 6},
    "L1":          {"label": "L1-Norm",  "color": "#2CA02C", "marker": "^", "ls": "-.", "z": 5},
    "Taylor":      {"label": "Taylor",   "color": "#9467BD", "marker": "D", "ls": ":", "z": 4},
    "FPGM":        {"label": "FPGM",     "color": "#8C564B", "marker": "v", "ls": "-", "z": 3},
    "HRank":       {"label": "HRank",    "color": "#17BECF", "marker": "P", "ls": "--", "z": 2},
}

# Build reverse lookup: alias -> canonical key
_ALIAS_TO_CANONICAL = {}
for canon, aliases in METHOD_ALIASES.items():
    for a in aliases:
        _ALIAS_TO_CANONICAL[a] = canon


def _canonicalize_method(name):
    """Map any method alias to its canonical key."""
    return _ALIAS_TO_CANONICAL.get(name, name)


def _parse_ratio_from_filename(name):
    """Extract target_ratio from filename pattern like arch_ds_method_rX.X_sN.json."""
    import re
    m = re.search(r"_r([\d.]+)_", name)
    return float(m.group(1)) if m else None


def _load_rows():
    rows = []
    seen = set()  # deduplicate by (arch, dataset, method, ratio, seed)
    for scan_dir in [RESULT_DIR, UNIFIED_DIR]:
        if not scan_dir.exists():
            continue
        for p in scan_dir.glob("*.json"):
            if p.name.endswith("_meta.json") or p.name.startswith("eb_"):
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "final_acc" not in d:
                continue
            # Ensure valid target_ratio: fallback to filename parsing
            if d.get("target_ratio") is None:
                d["target_ratio"] = _parse_ratio_from_filename(p.name)
            if d.get("target_ratio") is None or float(d["target_ratio"]) <= 0:
                continue
            d["method"] = _canonicalize_method(d.get("method", ""))
            # deduplicate: prefer chip_results over unified_results
            key = (d.get("architecture"), d.get("dataset"), d.get("method"),
                   str(d.get("target_ratio")), str(d.get("seed")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(d)
    return rows

def _load_eb_rows():
    rows = []
    for p in RESULT_DIR.glob("eb_*.json"):
        if p.name.endswith("_meta.json"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "final_acc" not in d:
            continue
        d["method"] = _canonicalize_method(d.get("method", ""))
        # EB files may lack target_ratio; parse from filename
        if d.get("target_ratio") is None:
            d["target_ratio"] = _parse_ratio_from_filename(p.name)
        if d.get("target_ratio") is None or float(d["target_ratio"]) <= 0:
            continue
        rows.append(d)
    return rows


def _check_data_completeness(rows, eb_rows):
    """Print warnings for missing data cells."""
    warnings = []
    for arch in ("resnet56", "vgg16"):
        for ratio in (0.7, 0.9):
            for method in METHOD_STYLE:
                n = sum(1 for r in rows
                        if r.get("architecture") == arch
                        and r.get("dataset") == "cifar100"
                        and r.get("method") == method
                        and abs(float(r.get("target_ratio", 0)) - ratio) < 0.01)
                if n == 0:
                    warnings.append(f"  MISSING: {arch}/{method}/r{ratio} (0 seeds)")
                elif n < 5:
                    warnings.append(f"  PARTIAL: {arch}/{method}/r{ratio} ({n}/5 seeds)")
    # EB completeness
    for arch in ("resnet56", "vgg16"):
        for ratio in (0.7, 0.9):
            for method in ("CHIP", "GRA-CHIP-v3", "L1"):
                n = sum(1 for r in eb_rows
                        if r.get("architecture") == arch
                        and r.get("method") == method
                        and abs(float(r.get("target_ratio", 0)) - ratio) < 0.01)
                if n == 0:
                    warnings.append(f"  MISSING EB: {arch}/{method}/r{ratio}")
    if warnings:
        print(f"DATA COMPLETENESS WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(w)
    else:
        print("Data completeness: ALL CELLS PRESENT")

def _rows(rows, arch, ds, method):
    out = []
    for r in rows:
        if r.get("architecture") == arch and r.get("dataset") == ds and r.get("method") == method:
            ratio = r.get("target_ratio")
            if ratio is None:
                continue
            out.append((float(ratio), int(r.get("seed", 0)), float(r["final_acc"])))
    return out

def _group_curve(rows, arch, ds, method):
    vals = _rows(rows, arch, ds, method)
    by_ratio = {}
    for ratio, _, acc in vals:
        by_ratio.setdefault(ratio, []).append(acc)
    xs = sorted(by_ratio)
    ys = [statistics.mean(by_ratio[x]) for x in xs]
    ystd = [statistics.stdev(by_ratio[x]) if len(by_ratio[x]) > 1 else 0.0 for x in xs]
    return xs, ys, ystd

def _paired_delta(rows, arch, ds, ratio, method_a, method_b):
    by_seed = {}
    for r in rows:
        if r.get("architecture") != arch or r.get("dataset") != ds:
            continue
        if float(r.get("target_ratio", 0)) != ratio:
            continue
        m = r.get("method")
        if m not in (method_a, method_b):
            continue
        s = int(r.get("seed", -1))
        by_seed.setdefault(s, {})[m] = float(r["final_acc"])
    return [by_seed[s][method_a] - by_seed[s][method_b]
            for s in sorted(by_seed)
            if method_a in by_seed[s] and method_b in by_seed[s]]

def _mean_ci95(vals):
    if not vals:
        return np.nan, np.nan
    mean = statistics.mean(vals)
    if len(vals) == 1:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(vals) / np.sqrt(len(vals))

def _eb_mean(eb_rows, arch, method, ratio):
    accs = [float(r["final_acc"]) for r in eb_rows
            if r.get("architecture") == arch
            and r.get("method") == method
            and abs(float(r.get("target_ratio", 0)) - ratio) < 0.01]
    if not accs:
        return np.nan, np.nan, 0
    m = statistics.mean(accs)
    s = statistics.stdev(accs) if len(accs) > 1 else 0.0
    return m, s, len(accs)


def _set_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "axes.facecolor": "#FAFBFC",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _draw_accuracy_panel(ax, rows, arch, ds, title, rng, show_ylabel=True):
    """Draw accuracy-vs-ratio curves for one architecture."""
    y_all = []
    for method, sty in METHOD_STYLE.items():
        xs, ys, ystd = _group_curve(rows, arch, ds, method)
        if not xs:
            continue
        xa, ya, sa = np.array(xs), np.array(ys), np.array(ystd)
        y_all.extend(list(zip(ys, [method] * len(ys))))

        ax.plot(xa, ya, ls=sty["ls"], marker=sty["marker"], color=sty["color"],
                lw=1.8, ms=5.5, label=sty["label"], zorder=sty["z"])
        # only show fill_between if std is reasonable (< 3pp) to avoid huge blobs
        for k in range(len(xa)):
            if sa[k] < 3.0:
                ax.fill_between(xa[k:k+2] if k < len(xa)-1 else xa[k:k+1],
                                (ya - sa)[k:k+2] if k < len(xa)-1 else (ya - sa)[k:k+1],
                                (ya + sa)[k:k+2] if k < len(xa)-1 else (ya + sa)[k:k+1],
                                color=sty["color"], alpha=0.10, lw=0, zorder=sty["z"] - 1)
        # seed jitter
        vals = _rows(rows, arch, ds, method)
        for x in xa:
            accs = [a for r, _, a in vals if r == x]
            if accs:
                xj = np.full(len(accs), x) + rng.normal(0, 0.003, len(accs))
                yj = np.array(accs) + rng.normal(0, 0.012, len(accs))
                ax.scatter(xj, yj, s=9, color=sty["color"], alpha=0.45,
                           edgecolors="white", lw=0.15, zorder=sty["z"] - 0.2)

    ratios = sorted(set(float(r.get("target_ratio", 0))
                        for r in rows if r.get("architecture") == arch and r.get("dataset") == ds))
    if ratios:
        ax.set_xticks(ratios)
        ax.set_xlim(min(ratios) - 0.04, max(ratios) + 0.04)

    # y-axis: clamp for VGG where L1 collapses
    y_vals = [v for v, _ in y_all]
    if y_vals:
        y_min, y_max = min(y_vals), max(y_vals)
        if y_min < 35:  # VGG L1 collapse
            y_low = 40.0
            ax.set_ylim(y_low, y_max + 2.0)
            # annotate clipped points
            for method, sty in METHOD_STYLE.items():
                vals = _rows(rows, arch, ds, method)
                for x in ratios:
                    accs = [a for r, _, a in vals if r == x]
                    if accs:
                        m = statistics.mean(accs)
                        if m < y_low:
                            ax.annotate("", xy=(x, y_low + 0.3), xytext=(x, y_low + 1.4),
                                        arrowprops=dict(arrowstyle="-|>", color=sty["color"], lw=1.2))
                            ax.text(x + 0.015, y_low + 1.6, f"{sty['label']}\n{m:.1f}%",
                                    color=sty["color"], fontsize=5.8, ha="center", va="bottom")
        else:
            pad = max(0.8, (y_max - y_min) * 0.18)
            ax.set_ylim(y_min - pad, y_max + pad)

    # highlight high-compression zone
    ax.axvspan(0.68, 0.94, color="#E6F0FF", alpha=0.45, zorder=0)

    # delta annotation box
    d07 = _paired_delta(rows, arch, ds, 0.7, "GRA-CHIP-v3", "CHIP")
    d09 = _paired_delta(rows, arch, ds, 0.9, "GRA-CHIP-v3", "CHIP")
    if d07 and d09:
        ax.text(0.03, 0.97,
                f"$\\Delta_{{0.7}}$={statistics.mean(d07):+.2f} pp\n$\\Delta_{{0.9}}$={statistics.mean(d09):+.2f} pp",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D1D5DB", alpha=0.92))

    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Pruning ratio")
    if show_ylabel:
        ax.set_ylabel("Top-1 Accuracy (%)")
    ax.grid(True, ls="--", lw=0.5, alpha=0.35)


def _draw_forest(ax, rows, baseline_method, baseline_label, cells, rng, xlim=None):
    """Draw a horizontal forest plot for paired deltas."""
    y_pos = np.arange(len(cells))[::-1]
    if xlim is None:
        xlim = (-1.0, 1.2)

    for i, (a, d, r, _lbl) in enumerate(cells):
        deltas = _paired_delta(rows, a, d, r, "GRA-CHIP-v3", baseline_method)
        mean, ci = _mean_ci95(deltas)
        y = y_pos[i]

        # seed-level scatter
        jitter = rng.normal(0, 0.06, len(deltas))
        ax.scatter(deltas, np.full(len(deltas), y) + jitter, s=18, color="#BFC7D5",
                   edgecolors="white", lw=0.3, alpha=0.85, zorder=2)

        # significance check
        sig = (not np.isnan(mean)) and (not np.isnan(ci)) and ((mean - ci > 0) or (mean + ci < 0))
        color = "#0B5FA5" if (sig and a == "resnet56") else ("#0F766E" if sig else "#6b7280")
        face = color if sig else "white"

        # clip mean and CI to axis limits
        plot_mean = float(np.clip(mean, xlim[0] + 0.03, xlim[1] - 0.03))
        lo = mean - ci
        hi = mean + ci
        clip_lo = max(lo, xlim[0])
        clip_hi = min(hi, xlim[1])

        ax.errorbar(plot_mean, y,
                     xerr=[[max(0, plot_mean - clip_lo)], [max(0, clip_hi - plot_mean)]],
                     fmt="o", color=color, ecolor=color, markerfacecolor=face,
                     elinewidth=1.8, capsize=3.2, ms=6, zorder=4)

        # off-chart arrow for clipped CI
        if hi > xlim[1] or lo < xlim[0]:
            edge = xlim[1] - 0.03 if hi > xlim[1] else xlim[0] + 0.03
            ax.annotate("", xy=(edge, y), xytext=(edge - 0.12 * (1 if hi > xlim[1] else -1), y),
                         arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.9), zorder=5)

        # value label
        ha = "left" if mean >= 0 else "right"
        nudge = 0.05 if mean >= 0 else -0.05
        ax.text(plot_mean + nudge, y + 0.24, f"{mean:+.2f}",
                fontsize=7.0, color=color, ha=ha, va="center", fontweight="bold")

    ax.axvline(0.0, color="#111827", ls="--", lw=0.9, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[3] for c in cells])
    ax.set_xlabel(f"$\\Delta$ (pp) GRA $-$ {baseline_label}")
    ax.set_xlim(*xlim)
    ax.grid(axis="x", ls="--", lw=0.5, alpha=0.35)


def generate_composite(rows, eb_rows):
    _set_style()
    fig = plt.figure(figsize=(15.0, 9.0))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.40, wspace=0.30,
                           left=0.055, right=0.985, top=0.925, bottom=0.065)
    rng = np.random.default_rng(20260302)

    cells = [
        ("resnet56", "cifar100", 0.7, "R56 p=0.7"),
        ("resnet56", "cifar100", 0.9, "R56 p=0.9"),
        ("vgg16", "cifar100", 0.7, "V16 p=0.7"),
        ("vgg16", "cifar100", 0.9, "V16 p=0.9"),
    ]

    # --- (a) ResNet-56 accuracy curves ---
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_accuracy_panel(ax_a, rows, "resnet56", "cifar100",
                         "(a) ResNet-56 / CIFAR-100", rng)

    # --- (b) VGG-16 accuracy curves ---
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_accuracy_panel(ax_b, rows, "vgg16", "cifar100",
                         "(b) VGG-16 / CIFAR-100", rng, show_ylabel=False)

    # --- (c) Forest plot: GRA vs CHIP ---
    ax_c = fig.add_subplot(gs[0, 2])
    _draw_forest(ax_c, rows, "CHIP", "CHIP", cells, rng, xlim=(-0.9, 1.1))
    ax_c.set_title("(c) Paired Effect vs CHIP", loc="left", fontweight="bold", pad=5)

    # --- (d) Iso-FLOPs grouped bar chart ---
    ax_d = fig.add_subplot(gs[1, 0])
    # Only show ResNet-56 (both ratios) in figure panel; VGG-16 EB data is in Table
    eb_cells_r56 = [
        ("resnet56", 0.7, "p = 0.7"),
        ("resnet56", 0.9, "p = 0.9"),
    ]
    eb_methods = [("CHIP", "#F28E2B", "CHIP"), ("GRA-CHIP-v3", "#0B5FA5", "GRA-CNN"), ("L1", "#2CA02C", "L1-Norm")]
    x_eb = np.arange(len(eb_cells_r56))
    bar_w = 0.22
    for j, (meth, col, lbl) in enumerate(eb_methods):
        means, stds = [], []
        for arch, ratio, _ in eb_cells_r56:
            m, s, _ = _eb_mean(eb_rows, arch, meth, ratio)
            means.append(m)
            stds.append(s)
        means_a = np.array(means, dtype=float)
        stds_a = np.array(stds, dtype=float)
        offset = (j - 1) * bar_w
        ax_d.bar(x_eb + offset, means_a, bar_w, yerr=stds_a, color=col, alpha=0.82,
                 capsize=3, ecolor="#555", label=lbl, zorder=3)
        for xi, mi in zip(x_eb + offset, means_a):
            if not np.isnan(mi):
                ax_d.text(xi, mi + 0.3, f"{mi:.1f}", ha="center", va="bottom",
                          fontsize=6.5, color=col, fontweight="bold")

    ax_d.set_xticks(x_eb)
    ax_d.set_xticklabels([c[2] for c in eb_cells_r56], fontsize=8)
    ax_d.set_ylabel("Top-1 Accuracy (%)")
    ax_d.set_title("(d) Iso-FLOPs on ResNet-56", loc="left", fontweight="bold", pad=5)
    ax_d.legend(fontsize=7.0, ncol=3, loc="upper right", frameon=False)
    ax_d.grid(axis="y", ls="--", lw=0.5, alpha=0.35)
    # tight y-axis
    all_eb_vals = []
    for meth, _, _ in eb_methods:
        for arch, ratio, _ in eb_cells_r56:
            m, _, _ = _eb_mean(eb_rows, arch, meth, ratio)
            if not np.isnan(m):
                all_eb_vals.append(m)
    if all_eb_vals:
        ax_d.set_ylim(min(all_eb_vals) - 2.5, max(all_eb_vals) + 2.5)

    # --- (e) Method comparison bar chart at p=0.7 ---
    ax_e = fig.add_subplot(gs[1, 1])
    # Use actual data from rows
    comp_methods = ["GRA-CHIP-v3", "CHIP", "L1", "Taylor", "FPGM", "HRank"]
    comp_labels  = ["GRA-CNN", "CHIP", "L1", "Taylor", "FPGM", "HRank"]
    comp_colors  = ["#0B5FA5", "#F28E2B", "#2CA02C", "#9467BD", "#8C564B", "#17BECF"]
    archs = [("resnet56", "R56"), ("vgg16", "V16")]
    x_comp = np.arange(len(archs))
    bw = 0.12
    for j, (meth, lbl, col) in enumerate(zip(comp_methods, comp_labels, comp_colors)):
        vals_m = []
        stds_m = []
        for arch, _ in archs:
            acc_list = [float(r["final_acc"]) for r in rows
                        if r.get("architecture") == arch
                        and r.get("dataset") == "cifar100"
                        and r.get("method") == meth
                        and abs(float(r.get("target_ratio", 0)) - 0.7) < 0.01]
            if acc_list:
                vals_m.append(statistics.mean(acc_list))
                stds_m.append(statistics.stdev(acc_list) if len(acc_list) > 1 else 0.0)
            else:
                vals_m.append(np.nan)
                stds_m.append(0.0)
        offset = (j - 2.5) * bw
        ax_e.bar(x_comp + offset, vals_m, bw, yerr=stds_m, color=col, alpha=0.82,
                 capsize=2, ecolor="#777", label=lbl, zorder=3)
        for xi, vi in zip(x_comp + offset, vals_m):
            if not np.isnan(vi):
                ax_e.text(xi, vi + 0.15, f"{vi:.1f}", ha="center", va="bottom",
                          fontsize=5.2, color=col, rotation=90)

    ax_e.set_xticks(x_comp)
    ax_e.set_xticklabels([a[1] for a in archs], fontsize=8)
    ax_e.set_ylabel("Top-1 Accuracy (%)")
    ax_e.set_title("(e) All Methods at p=0.7", loc="left", fontweight="bold", pad=5)
    ax_e.legend(fontsize=5.8, ncol=3, loc="lower left", frameon=False, borderpad=0.3)
    ax_e.grid(axis="y", ls="--", lw=0.5, alpha=0.35)
    # y-axis: focus on the relevant range
    all_comp = []
    for meth in comp_methods:
        for arch, _ in archs:
            acc_list = [float(r["final_acc"]) for r in rows
                        if r.get("architecture") == arch
                        and r.get("dataset") == "cifar100"
                        and r.get("method") == meth
                        and abs(float(r.get("target_ratio", 0)) - 0.7) < 0.01]
            if acc_list:
                all_comp.append(statistics.mean(acc_list))
    if all_comp:
        ax_e.set_ylim(min(all_comp) - 2.0, max(all_comp) + 2.5)

    # --- (f) Forest plot: GRA vs L1 ---
    ax_f = fig.add_subplot(gs[1, 2])
    _draw_forest(ax_f, rows, "L1", "L1", cells, rng, xlim=(-2.0, 2.5))
    ax_f.set_title("(f) Paired Effect vs L1", loc="left", fontweight="bold", pad=5)

    # --- shared legend for curves (a)(b) ---
    handles_ab, labels_ab = ax_a.get_legend_handles_labels()
    fig.legend(handles_ab, labels_ab, ncol=6, loc="upper center", frameon=False,
               bbox_to_anchor=(0.36, 0.998), handlelength=1.8, columnspacing=0.9, fontsize=7.8)

    fig.savefig(OUT_DIR / "fig_composite_results.pdf", format="pdf")
    fig.savefig(OUT_DIR / "fig_composite_results.png", dpi=600)
    print(f"Saved: {OUT_DIR / 'fig_composite_results.pdf'}")
    print(f"Saved: {OUT_DIR / 'fig_composite_results.png'}")


if __name__ == "__main__":
    rows = _load_rows()
    eb_rows = _load_eb_rows()
    print(f"Loaded {len(rows)} main results, {len(eb_rows)} equal-budget results")
    _check_data_completeness(rows, eb_rows)
    generate_composite(rows, eb_rows)

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PKG = ROOT / "submission_package_final"
PKG_FIG = PKG / "figures"
PKG_TAB = PKG / "tables"

FIGURES = [
    "fig_pipeline.pdf",
    "fig_pipeline.png",
    "fig_main_ratio_curves.pdf",
    "fig_main_ratio_curves.png",
    "fig_ablation_extended.pdf",
    "fig_ablation_extended.png",
    "fig_semantic_evidence.pdf",
    "fig_semantic_evidence.png",
    "fig_tiny_matrix.pdf",
    "fig_tiny_matrix.png",
    "fig_tiny_ratio_curve.pdf",
    "fig_tiny_ratio_curve.png",
    "fig_dense_ratio_extension.pdf",
    "fig_dense_ratio_extension.png",
    "fig_dense_delta_curves.pdf",
    "fig_dense_delta_curves.png",
    "fig_dense_rank_heatmap.pdf",
    "fig_dense_rank_heatmap.png",
    "fig_sensitivity_multiregime.pdf",
    "fig_sensitivity_multiregime.png",
    "fig_sensitivity_heatmap.pdf",
    "fig_sensitivity_heatmap.png",
]

TABLES = [
    "primary_results_ivc.tex",
    "paired_robustness_ivc.tex",
    "equal_budget_ivc.tex",
    "ablation_summary_ivc.tex",
    "full_cifar_matrix_ivc.tex",
    "residual_paired_full_ivc.tex",
    "metric_sensitivity_ivc.tex",
    "cross_dataset_tiny.tex",
    "runtime_profile.tex",
    "latency_profile_local.tex",
    "dense_ratio_extension_local.tex",
    "sensitivity_multiregime_local.tex",
]


def copy2(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True, exist_ok=True)
    PKG_FIG.mkdir(parents=True, exist_ok=True)
    PKG_TAB.mkdir(parents=True, exist_ok=True)

    copy2(ROOT / "manuscript_ivc.tex", PKG / "manuscript.tex")
    copy2(ROOT / "manuscript_ivc.pdf", PKG / "manuscript.pdf")
    copy2(ROOT / "Responses_to_Technical_Check_Results.pdf", PKG / "Responses_to_Technical_Check_Results.pdf")
    copy2(ROOT / "Responses_to_Technical_Check_Results.txt", PKG / "Responses_to_Technical_Check_Results.txt")
    copy2(ROOT / "abstract_ivc.pdf", PKG / "abstract.pdf")
    copy2(ROOT / "abstract.txt", PKG / "abstract.txt")
    copy2(ROOT / "supplement_ivc.pdf", PKG / "supplement.pdf")
    copy2(ROOT / "cover_letter_ivc.pdf", PKG / "cover_letter.pdf")
    copy2(ROOT / "highlights.txt", PKG / "highlights.txt")
    copy2(ROOT / "highlights_ivc.pdf", PKG / "highlights.pdf")
    copy2(ROOT / "graphical_abstract.png", PKG / "graphical_abstract.png")
    copy2(ROOT / "graphical_abstract.pdf", PKG / "graphical_abstract.pdf")
    copy2(ROOT / "references.bib", PKG / "references.bib")
    copy2(ROOT / "elsarticle.cls", PKG / "elsarticle.cls")
    copy2(ROOT / "elsarticle-num.bst", PKG / "elsarticle-num.bst")
    copy2(ROOT / "SUBMISSION_GUIDE_IVC.md", PKG / "SUBMISSION_GUIDE_IVC.md")
    copy2(ROOT / "FINAL_SUBMISSION_CHECKLIST.md", PKG / "FINAL_SUBMISSION_CHECKLIST.md")
    copy2(ROOT / "reviewer_shortlist.md", PKG / "reviewer_shortlist.md")

    for name in FIGURES:
        copy2(ROOT / "figures" / name, PKG_FIG / name)
    for name in TABLES:
        copy2(ROOT / "tables" / name, PKG_TAB / name)

    print("submission package assembled at", PKG)


if __name__ == "__main__":
    main()

# GRA-CNN Agent Rules

Date: 2026-05-28

## Source Priority

Use repository evidence in this order:

1. `submission/ivc_final/manuscript.tex` and `submission/ivc_final/manuscript.pdf`
2. `experiments/run_chip_worker.py`, especially `score_gra_chip_v3`
3. `paper/tables/`, `paper/tables/source_csv/`, and `results/`
4. `paper/source_tools/` for figure/table regeneration context
5. `docs/handoff/PROJECT_MEMORY.md` for project state

## Writing Rules

- Do not fabricate references, numbers, datasets, or experimental conclusions.
- Do not claim GRA-CNN is best in every setting; preserve the paper's nuance.
- Treat VGG results as architecture-dependent.
- Describe semantic refinement as a boundary correction, not a full replacement
  for structural pruning.
- Keep the method name as `GRA-CNN` for the paper-derived method unless a patent
  title needs a more general technical name.
- Do not cite obsolete previous-journal package names or old local IVC v1/v2 paths.
- Use the latest MoE patent document only as a style benchmark if available
  locally; do not import MoE technical details.

## Engineering Rules

- Avoid launching long experiments unless explicitly requested.
- Keep raw datasets and checkpoints outside Git.
- Keep generated logs, lock files, LaTeX temporary files, and local environment
  files outside Git.
- Before pushing, check for stale paths, files over 100 MB, raw datasets, and
  checkpoints.
- Prefer small, evidence-preserving edits over broad rewrites.

## Patent Drafting Rules

- First extract invention points and technical effects from the manuscript and
  code.
- Then draft the disclosure in Chinese patent style: background problem,
  technical solution, optional formulas, embodiments, and beneficial effects.
- Build claims around method steps, device/system modules, storage medium, and
  optional training/pruning pipeline embodiments.
- Make formulas traceable to the method description; define every symbol.
- Use figures that explain the pruning pipeline, boundary refinement, and
  class-aware GRA redundancy.

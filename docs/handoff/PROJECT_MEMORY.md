# GRA-CNN Project Memory

Date: 2026-05-28

## Current Project State

This repository is the curated GitHub handoff for the GRA-CNN paper and patent
writing work. It keeps the complete final IVC submission package, reproducible
code, filtered result data, and the cross-machine rules/prompt needed for
patent drafting.

Canonical final submission package:

- `submission/ivc_final/`

Canonical paper assets:

- `paper/manuscript.pdf`
- `paper/supplement.pdf`
- `paper/figures/`
- `paper/tables/`
- `paper/tables/source_csv/`
- `paper/source_tools/`

Canonical implementation:

- public entrypoint: `experiments/run_real_pruning.py`
- implementation worker: `experiments/run_chip_worker.py`
- submitted method: `GRA-CNN`
- core function: `score_gra_chip_v3`

## Technical Core

GRA-CNN is a semantic-aware structured channel pruning method. The submitted
method should be described around these invention points:

1. A structural anchor combines channel independence, Taylor sensitivity, and
   Fisher-style information to produce a stable channel ranking.
2. A class-aware GRA redundancy estimate measures semantic replaceability from
   class-conditioned activation trends.
3. A semantic quality gate estimates whether the semantic signal is reliable
   enough for a layer.
4. Semantic refinement is applied only near the keep/prune boundary, so the
   semantic signal corrects borderline decisions instead of replacing the
   structural ranking globally.
5. The final keep masks are used for dependency-safe structured channel pruning
   and model reconstruction.

Use these phrases consistently:

- semantic-aware structured channel pruning
- Gray Relational Analysis
- class-aware semantic redundancy
- structural anchor
- quality-gated semantic boundary refinement
- keep/prune boundary

## Evidence And Results

The tracked `results/` directory contains filtered seed-level JSON records and
derived CSV/LaTeX tables used by the submitted paper. The strongest evidence is
for high-compression residual CNN settings. VGG-style behavior is more
architecture-dependent and should not be overstated.

Use `paper/tables/` and `paper/tables/source_csv/` for paper-level numeric
evidence. Do not invent extra experiments or performance numbers.

## Excluded Heavy Artifacts

The repository intentionally does not include:

- raw CIFAR or Tiny-ImageNet dataset files
- pretrained `.pth`, `.pt`, or `.ckpt` checkpoints
- raw launcher logs and lock files
- obsolete previous-journal submission folders
- old local IVC v1/v2 technical-check folders

Dataset and checkpoint layouts are documented in `data/README.md` and
`checkpoints/README.md`.

## Patent-Writing Direction

The next machine should load the local `patent-disclosure-skill` and use
`docs/handoff/GRA_PATENT_WRITING_PROMPT.md` as the starting prompt. If a recent
MoE patent document is available locally, use it only as a language/style
benchmark. Do not copy MoE technical content into the GRA-CNN patent.

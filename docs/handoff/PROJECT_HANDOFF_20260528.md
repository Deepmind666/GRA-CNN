# Project Handoff 2026-05-28

## Purpose

Prepare the GRA-CNN GitHub `main` branch for use on another computer for patent
writing, while freeing the original local workstation for the MoE project.

## What Is Included

- complete extracted final IVC submission package: `submission/ivc_final/`
- public reproducibility code: `experiments/`, `models/`, `pruning/`
- paper-level evidence and filtered run data: `paper/`, `results/`
- dataset and checkpoint layout notes: `data/README.md`,
  `checkpoints/README.md`
- patent handoff memory, rules, and prompt: `docs/handoff/`

## What Was Intentionally Excluded

- raw public datasets
- pretrained checkpoint binaries
- old previous-journal submission folders
- old IVC v1/v2 technical-check folders
- raw logs, lock files, temporary LaTeX products, and workstation diagnostics

## Next Session Checklist

1. Clone or pull GitHub `main` on the new computer.
2. Read `docs/handoff/PROJECT_MEMORY.md`.
3. Read `docs/handoff/AGENT_RULES.md`.
4. Start patent drafting with `docs/handoff/GRA_PATENT_WRITING_PROMPT.md`.
5. Load the local `patent-disclosure-skill`.
6. Use the latest MoE patent document only as a language/style benchmark.

## Notes

The original local working directory was not cleaned in place.
GitHub `main` is the curated handoff source.


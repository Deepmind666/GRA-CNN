# Codex Rules For GRA-CNN

Use `docs/handoff/PROJECT_MEMORY.md` and `docs/handoff/AGENT_RULES.md` as the
authoritative project memory and operating rules.

For patent writing, start from `docs/handoff/GRA_PATENT_WRITING_PROMPT.md` and
use the local `patent-disclosure-skill`. Treat the latest MoE patent document
only as a language-quality benchmark; do not copy MoE technical content.

Canonical package and evidence:

- final submission package: `submission/ivc_final/`
- paper assets: `paper/`
- run-level result data: `results/`
- implementation entrypoint: `experiments/run_real_pruning.py`
- implementation worker: `experiments/run_chip_worker.py`
- core GRA-CNN scoring code: `pruning/gra_chip.py`

Do not commit raw datasets, checkpoints, local paths, virtual environments,
launcher logs, or LaTeX build artifacts.

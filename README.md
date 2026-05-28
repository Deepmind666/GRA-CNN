# GRA-CNN: Semantic-Aware Structured Channel Pruning via Gray Relational Analysis

This repository provides the public research release for **GRA-CNN**, a structured channel pruning framework that combines a stable structural anchor with a quality-gated, class-aware semantic refinement based on Gray Relational Analysis (GRA).

The repository matches the version submitted to **Image and Vision Computing** in March 2026 and is organized as a reviewer-facing release: core code, paper assets, and filtered run-level results are included; private notes, raw launcher logs, local environment files, and heavyweight training artifacts are intentionally excluded.

## Authors

- Kangrui Li
- Junyi Lin
- Xiaobo Zhang

## Contact

For code, reproducibility, repository maintenance, and implementation-related
questions, please contact:

**Kangrui Li**  
**Email:** 1403073295@mails.gdut.edu.cn

For formal manuscript-related correspondence, please refer to the
corresponding-author information provided in the paper.

## Method Summary

GRA-CNN is designed for structured pruning under aggressive compression, where purely structural rankings can become brittle.

The public implementation exposes three main ideas:

1. **Structural anchor**
   - combines channel independence, Taylor sensitivity, and Fisher-style signals for stable global ranking
2. **Class-aware semantic redundancy**
   - uses class-conditioned activation trends to estimate whether a channel is replaceable
3. **Quality-gated boundary refinement**
   - applies semantic correction only near the keep/prune boundary when the semantic signal is sufficiently reliable

The strongest gains in the submitted study appear in **high-compression residual CNN settings**, while behavior on VGG-style networks is more architecture-dependent.

## Repository Layout

```text
models/       CNN backbones used in the experiments
pruning/      GRA-CNN scoring logic, structural baselines, and pruning utilities
experiments/  Public experiment entrypoints and benchmark utilities
paper/        Submitted manuscript PDF, supplement PDF, and paper figures/tables
results/      Filtered run-level JSON records and derived result tables
submission/   Complete extracted IVC submission package
docs/handoff/ Patent-writing memory, rules, and cross-machine handoff prompt
data/         Dataset layout note; raw datasets are intentionally untracked
```

## Environment

The public release uses Python 3.10+ and PyTorch.

Install the base dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

The code expects datasets under a local `data/` directory by default.
CIFAR-10 and CIFAR-100 are downloaded by `torchvision` when the runner is used.
Tiny-ImageNet must be prepared locally.

Recommended layout:

```text
data/
  cifar-10-batches-py/
  cifar-100-python/
  tiny-imagenet-200/
    train/
    val/
```

For Tiny-ImageNet, scripts also accept an explicit `--data-dir` argument.
See `data/README.md` for the expected layout.

## Checkpoints

Pretrained baseline checkpoints are **not bundled** in this public release because the repository is intended to stay lightweight and portable.

Expected checkpoint locations are documented in:

```text
checkpoints/
```

You should place local baseline weights there before running the full pruning pipeline.

## Minimal Reproduction Commands

Inspect the public experiment CLI:

```bash
python experiments/run_real_pruning.py --help
```

`experiments/run_real_pruning.py` delegates to `experiments/run_chip_worker.py`.
The submitted `GRA-CNN` method is implemented by `score_gra_chip_v3` in that
worker and combines the structural anchor, class-aware GRA redundancy, and
quality-gated keep/prune boundary refinement.

Example GRA-CNN run:

```bash
python experiments/run_real_pruning.py --arch resnet56 --dataset cifar100 --method GRA-CNN --target_ratio 0.7 --seed 42 --epochs 30
```

Example metric-sensitivity run:

```bash
python experiments/run_metric_ablation_worker.py ^
  --arch resnet56 ^
  --dataset cifar100 ^
  --method Cosine-Semantic ^
  --ratio 0.7 ^
  --seed 42 ^
  --finetune_epochs 40
```

Latency benchmark entrypoint:

```bash
python experiments/benchmark_latency.py
```

## Paper Assets

The submitted paper artifacts are included in:

- `paper/manuscript.pdf`
- `paper/supplement.pdf`

The accompanying publication figures and tables are stored under:

- `paper/figures/`
- `paper/tables/`
- `paper/source_tools/`

The complete extracted final IVC submission package is included in:

- `submission/ivc_final/`

This is the canonical package for portal upload checks and patent-writing
source review.

## Results

The `results/` directory contains filtered run-level outputs used to support the paper:

- seed-level JSON records
- derived CSV tables
- selected LaTeX result tables
- a small number of analysis summaries

Raw launcher logs, lock files, and internal go/no-go notes are intentionally excluded from the public release.

## Patent Handoff

Cross-machine handoff material for patent drafting is stored in:

- `docs/handoff/PROJECT_MEMORY.md`
- `docs/handoff/AGENT_RULES.md`
- `docs/handoff/GRA_PATENT_WRITING_PROMPT.md`
- `docs/handoff/PROJECT_HANDOFF_20260528.md`

Use these files together with the local `patent-disclosure-skill` on the new
machine. The handoff prompt asks the patent writer to match the language level
of the latest MoE patent document as a style reference only; it must not import
MoE technical content into the GRA-CNN patent.

## Citation

If you use this repository, please cite the manuscript as:

```bibtex
@misc{li2026gracnn,
  title        = {GRA-CNN: Semantic-Aware Structured Channel Pruning via Gray Relational Analysis},
  author       = {Kangrui Li and Junyi Lin and Xiaobo Zhang},
  year         = {2026},
  note         = {Submitted to Image and Vision Computing}
}
```

## License

This repository is released under the **MIT License**. See `LICENSE` for details.

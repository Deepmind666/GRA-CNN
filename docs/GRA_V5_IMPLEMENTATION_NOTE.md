# GRA v5 Implementation Note

## Why v5
P1 results show v4 has mixed behavior:
- Good at low pruning ratio (r=0.3) but unstable in medium/high settings.
- Holm-Bonferroni corrected tests report 0 significant wins and 3 significant losses.

Goal of v5: keep semantic benefit while reducing instability from noisy semantic scores.

## New files
- C:\GRA-CNN\pruning\core_algorithm_v5_improved.py
- C:\GRA-CNN\experiments\run_full_matrix_v5.py
- C:\GRA-CNN\experiments\single_task_v5.py

## v5 algorithm changes
1. Reliability-gated semantic fusion
- Estimate semantic reliability per layer using split-half Spearman agreement.
- If reliability is low, automatically down-weight GRA and back off to L1/Fisher.

2. Robust margin reference
- Smooth margin reference with standardized-and-clipped transform + sigmoid.
- Prevents extreme margins from dominating channel scoring.

3. Robust score normalization
- Replace naive min-max with quantile normalization.
- Reduces outlier sensitivity in Fisher/L1/GRA/Ortho components.

4. Ratio-aware and depth-aware adaptive weights
- Piecewise base weights for low/medium/high pruning ratios.
- Depth shift still supported but bounded, then reliability gate is applied.

## Engineering quality improvements
- New code keeps v4 code untouched.
- Type hints and stricter guardrails are added.
- Short-run finetune guard in v5 runner: if epochs < 10, return real evaluated accuracy instead of 0.0.

## Smoke test (local)
Command:
- python experiments/single_task_v5.py --arch resnet20 --method GRA-v5 --ratio 0.3 --seed 42 --pruning_scope stage_mid_only --finetune_epochs 1

Observed output (RESULT_JSON):
- gra_version: 5.0
- baseline_acc: 80.62
- pruned_acc: 42.96
- final_acc: 84.83

## Recommended first validation matrix
Use same P1 matrix for comparability:
- architectures: resnet20, resnet56
- methods: L1, FPGM, GRA-v4, GRA-v5, Random
- ratios: 0.3, 0.5, 0.7
- seeds: 42, 123, 456, 789, 1024

## Acceptance targets for v5
- At least reduce corrected significant losses vs v4 baseline (from 3 to <=1).
- Improve mean rank versus v4 in medium/high ratios.
- Keep evidence CSV fields unchanged and complete.

"""Single-task launcher for GRA v5 experiments."""
import sys
sys.path.insert(0, r"C:\GRA-CNN")

import argparse
import json
import torch

from experiments.run_full_matrix_v5 import run_single_experiment_v5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--finetune_epochs", type=int, default=40)
    parser.add_argument(
        "--pruning_scope",
        default="stage_mid_only",
        choices=["resnet_conv1_only", "stage_mid_only"],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_single_experiment_v5(
        arch=args.arch,
        method=args.method,
        ratio=args.ratio,
        seed=args.seed,
        device=device,
        pruning_scope=args.pruning_scope,
        dataset=args.dataset,
        finetune_epochs=args.finetune_epochs,
    )
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))

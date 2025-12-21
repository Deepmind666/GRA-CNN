@echo off
REM ImageNet-100 Experiment Runner for GRA-CNN
REM Runs both GRA and L1 pruning at multiple ratios

echo ============================================
echo GRA-CNN ImageNet-100 Experiment
echo ============================================

cd /d C:\GRA-CNN

REM Activate conda environment if needed
REM call conda activate pyt-sm12-build

echo.
echo [1/6] Running GRA Pruning (ratio=0.3)...
python experiments/imagenet_experiment.py --method gra --prune-ratio 0.3 --mock --finetune-epochs 5

echo.
echo [2/6] Running GRA Pruning (ratio=0.5)...
python experiments/imagenet_experiment.py --method gra --prune-ratio 0.5 --mock --finetune-epochs 5

echo.
echo [3/6] Running GRA Pruning (ratio=0.7)...
python experiments/imagenet_experiment.py --method gra --prune-ratio 0.7 --mock --finetune-epochs 5

echo.
echo [4/6] Running L1 Pruning (ratio=0.3)...
python experiments/imagenet_experiment.py --method l1 --prune-ratio 0.3 --mock --finetune-epochs 5

echo.
echo [5/6] Running L1 Pruning (ratio=0.5)...
python experiments/imagenet_experiment.py --method l1 --prune-ratio 0.5 --mock --finetune-epochs 5

echo.
echo [6/6] Running L1 Pruning (ratio=0.7)...
python experiments/imagenet_experiment.py --method l1 --prune-ratio 0.7 --mock --finetune-epochs 5

echo.
echo ============================================
echo All experiments completed!
echo Results saved to experiments/imagenet/
echo ============================================

pause

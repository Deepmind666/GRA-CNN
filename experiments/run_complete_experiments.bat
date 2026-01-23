@echo off
REM ============================================
REM GRA-CNN 完整实验批量运行脚本
REM ============================================
REM 使用说明：
REM 1. 确保 GPU 空闲
REM 2. 激活 gra311 环境
REM 3. 运行此脚本
REM ============================================

echo ============================================
echo GRA-CNN 完整实验矩阵
echo ============================================

call conda activate gra311

echo [1/12] ResNet-20 / CIFAR-10 / GRA / 0.5
python experiments\run_real_pruning.py --arch resnet20 --dataset cifar10 --method gra --ratio 0.5 --epochs 40

echo [2/12] ResNet-20 / CIFAR-10 / L1 / 0.5
python experiments\run_real_pruning.py --arch resnet20 --dataset cifar10 --method l1 --ratio 0.5 --epochs 40

echo [3/12] ResNet-56 / CIFAR-10 / GRA / 0.5
python experiments\run_real_pruning.py --arch resnet56 --dataset cifar10 --method gra --ratio 0.5 --epochs 40

echo [4/12] ResNet-56 / CIFAR-10 / L1 / 0.5
python experiments\run_real_pruning.py --arch resnet56 --dataset cifar10 --method l1 --ratio 0.5 --epochs 40

echo [5/12] ResNet-56 / CIFAR-10 / FPGM / 0.5
python experiments\run_real_pruning.py --arch resnet56 --dataset cifar10 --method fpgm --ratio 0.5 --epochs 40

echo [6/12] ResNet-56 / CIFAR-10 / HRank / 0.5
python experiments\run_real_pruning.py --arch resnet56 --dataset cifar10 --method hrank --ratio 0.5 --epochs 40

echo [7/12] ResNet-56 / CIFAR-100 / GRA / 0.5
python experiments\run_real_pruning.py --arch resnet56 --dataset cifar100 --method gra --ratio 0.5 --epochs 40

echo [8/12] ResNet-56 / CIFAR-100 / L1 / 0.5
python experiments\run_real_pruning.py --arch resnet56 --dataset cifar100 --method l1 --ratio 0.5 --epochs 40

echo [9/12] ResNet-110 / CIFAR-10 / GRA / 0.5
python experiments\run_real_pruning.py --arch resnet110 --dataset cifar10 --method gra --ratio 0.5 --epochs 40

echo [10/12] ResNet-110 / CIFAR-10 / L1 / 0.5
python experiments\run_real_pruning.py --arch resnet110 --dataset cifar10 --method l1 --ratio 0.5 --epochs 40

echo [11/12] VGG-16 / CIFAR-10 / GRA / 0.5
python experiments\run_real_pruning.py --arch vgg16 --dataset cifar10 --method gra --ratio 0.5 --epochs 40

echo [12/12] VGG-16 / CIFAR-10 / L1 / 0.5
python experiments\run_real_pruning.py --arch vgg16 --dataset cifar10 --method l1 --ratio 0.5 --epochs 40

echo ============================================
echo 所有实验完成！结果保存在：
echo experiments\supplementary_results.csv
echo ============================================
pause

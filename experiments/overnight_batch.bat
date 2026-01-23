@echo off
echo Starting GRA-Fisher 10-hour Deep Optimization Experiments...
echo.

set PYTHON=C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe
set SCRIPT=C:\GRA-CNN\experiments\run_real_pruning.py

REM Experiment 1
echo [1/23] ResNet-56/CIFAR-10 @ 0.5
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

REM Experiment 2
echo [2/23] ResNet-56/CIFAR-10 @ 0.3
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-10 --method GRA --ratio 0.3 --epochs 40

REM Experiment 3
echo [3/23] ResNet-56/CIFAR-10 @ 0.7
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-10 --method GRA --ratio 0.7 --epochs 40

REM Experiment 4
echo [4/23] VGG-16/CIFAR-10 @ 0.5
%PYTHON% %SCRIPT% --arch VGG-16 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

REM Experiment 5
echo [5/23] ResNet-20/CIFAR-10 @ 0.5
%PYTHON% %SCRIPT% --arch ResNet-20 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

REM Experiment 6
echo [6/23] ResNet-110/CIFAR-10 @ 0.5
%PYTHON% %SCRIPT% --arch ResNet-110 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

REM Experiment 7
echo [7/23] ResNet-56/CIFAR-100 @ 0.5
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-100 --method GRA --ratio 0.5 --epochs 40

REM Experiment 8 - Replicate for stability
echo [8/23] ResNet-56/CIFAR-10 @ 0.5 (repeat)
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

REM Experiment 9
echo [9/23] ResNet-56/CIFAR-10 @ 0.4
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-10 --method GRA --ratio 0.4 --epochs 40

REM Experiment 10
echo [10/23] ResNet-56/CIFAR-10 @ 0.6
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-10 --method GRA --ratio 0.6 --epochs 40

REM Experiment 11
echo [11/23] VGG-16/CIFAR-10 @ 0.3
%PYTHON% %SCRIPT% --arch VGG-16 --dataset CIFAR-10 --method GRA --ratio 0.3 --epochs 40

REM Experiment 12
echo [12/23] VGG-16/CIFAR-10 @ 0.7
%PYTHON% %SCRIPT% --arch VGG-16 --dataset CIFAR-10 --method GRA --ratio 0.7 --epochs 40

REM Experiment 13
echo [13/23] ResNet-20/CIFAR-10 @ 0.3
%PYTHON% %SCRIPT% --arch ResNet-20 --dataset CIFAR-10 --method GRA --ratio 0.3 --epochs 40

REM Experiment 14
echo [14/23] ResNet-20/CIFAR-10 @ 0.7
%PYTHON% %SCRIPT% --arch ResNet-20 --dataset CIFAR-10 --method GRA --ratio 0.7 --epochs 40

REM Experiment 15
echo [15/23] ResNet-110/CIFAR-10 @ 0.3
%PYTHON% %SCRIPT% --arch ResNet-110 --dataset CIFAR-10 --method GRA --ratio 0.3 --epochs 40

REM Experiment 16
echo [16/23] ResNet-110/CIFAR-10 @ 0.7
%PYTHON% %SCRIPT% --arch ResNet-110 --dataset CIFAR-10 --method GRA --ratio 0.7 --epochs 40

REM Experiment 17
echo [17/23] ResNet-56/CIFAR-100 @ 0.3
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-100 --method GRA --ratio 0.3 --epochs 40

REM Experiment 18
echo [18/23] ResNet-56/CIFAR-100 @ 0.7
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-100 --method GRA --ratio 0.7 --epochs 40

REM Experiment 19
echo [19/23] VGG-16/CIFAR-100 @ 0.5
%PYTHON% %SCRIPT% --arch VGG-16 --dataset CIFAR-100 --method GRA --ratio 0.5 --epochs 40

REM Experiment 20 - Final replications
echo [20/23] ResNet-56/CIFAR-10 @ 0.5 (repeat 2)
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

REM Experiment 21
echo [21/23] VGG-16/CIFAR-10 @ 0.5 (repeat)
%PYTHON% %SCRIPT% --arch VGG-16 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

REM Experiment 22
echo [22/23] ResNet-110/CIFAR-10 @ 0.5 (repeat)
%PYTHON% %SCRIPT% --arch ResNet-110 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

REM Experiment 23
echo [23/23] ResNet-56/CIFAR-10 @ 0.5 (repeat 3)
%PYTHON% %SCRIPT% --arch ResNet-56 --dataset CIFAR-10 --method GRA --ratio 0.5 --epochs 40

echo.
echo ============================================
echo All experiments completed!
echo ============================================

@echo off
REM Comprehensive GRA-CNN Experiment Runner
REM Runs full experiment matrix in background

set PYTHON=C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe
set SCRIPT=C:\GRA-CNN\experiments\run_comprehensive.py
set LOG_DIR=C:\GRA-CNN\experiments\logs

mkdir %LOG_DIR% 2>nul

echo Starting comprehensive experiments...
echo Logging to %LOG_DIR%

REM Phase 1: ResNet-56 CIFAR-10 (Priority)
echo [Phase 1] ResNet-56 CIFAR-10
%PYTHON% %SCRIPT% --archs resnet56 --datasets cifar10 --methods l1 fpgm hrank gra --ratios 0.3 0.5 0.7 --seeds 0 1 2 --save-dir C:\GRA-CNN\experiments\comprehensive > %LOG_DIR%\phase1.log 2>&1

REM Phase 2: ResNet-56 CIFAR-100
echo [Phase 2] ResNet-56 CIFAR-100
%PYTHON% %SCRIPT% --archs resnet56 --datasets cifar100 --methods l1 fpgm hrank gra --ratios 0.3 0.5 0.7 --seeds 0 1 2 --save-dir C:\GRA-CNN\experiments\comprehensive > %LOG_DIR%\phase2.log 2>&1

REM Phase 3: ResNet-20 CIFAR-10/100
echo [Phase 3] ResNet-20 CIFAR-10/100
%PYTHON% %SCRIPT% --archs resnet20 --datasets cifar10 cifar100 --methods l1 gra --ratios 0.3 0.5 0.7 --seeds 0 1 2 --save-dir C:\GRA-CNN\experiments\comprehensive > %LOG_DIR%\phase3.log 2>&1

REM Phase 4: ResNet-110 CIFAR-10
echo [Phase 4] ResNet-110 CIFAR-10
%PYTHON% %SCRIPT% --archs resnet110 --datasets cifar10 --methods l1 gra --ratios 0.3 0.5 0.7 --seeds 0 1 2 --save-dir C:\GRA-CNN\experiments\comprehensive > %LOG_DIR%\phase4.log 2>&1

echo All experiments completed!
echo Results saved to: C:\GRA-CNN\experiments\comprehensive

# GRA-CNN v5 双端执行工作流规范

## 1. 目标
- 本地（Codex）与服务器（Claude）并行跑 v5 实验，缩短总 wall-clock 时间。
- 两端运行时长尽可能接近，避免一端长时间空转。
- 全流程可恢复、可追踪、可复核。

## 2. 当前基线与资源
- 本地路径: c:\GRA-CNN
- 服务器路径: C:\Users\sshuser\GRA-CNN\project
- 服务器连接: ssh FatMachine
- 服务器 GPU: RTX 5090 D v2, 24455 MiB
- 服务器 Conda 环境: aether-wsn (Python 3.13.11, torch 2.6.0+cu124)

## 3. 已修正的脚本问题
- experiments/single_task_v5.py
  - 修复为相对项目根路径导入，不再写死 C:\GRA-CNN。
- experiments/stage_mid_pruning/run_p1_validation_v5.py
  - 修复为可移植路径。
  - 支持可配置参数: --python_exe --single_task_script --result_dir --timeout_sec 等。
  - 默认 timeout 5400 秒，避免 1800 秒误超时。

## 4. 分工与等时长拆分
依据（2026-02-09 实测）:
- 本地 resnet20|L1|r=0.3|seed=42: 404 秒
- 本地 resnet56|L1|r=0.3|seed=42: 412 秒
- 历史 run_log(resnet56) 平均: 443 秒

拆分方案（接近等时长）:
- Codex 本地: 跑 resnet20 全矩阵
  - architectures=resnet20
  - methods=L1,FPGM,GRA-v4,GRA-v5,Random
  - ratios=0.3,0.5,0.7
  - seeds=42,123,456,789,1024
  - 任务数=75
- Claude 服务器: 跑 resnet56 全矩阵
  - architectures=resnet56
  - methods=L1,FPGM,GRA-v4,GRA-v5,Random
  - ratios=0.3,0.5,0.7
  - seeds=42,123,456,789,1024
  - 任务数=75

ETA 估算:
- 单任务约 408~443 秒
- 每端 75 任务约 8.5~9.2 小时
- 依据: 上述实测与历史 run_log

## 5. 启动命令（防断线）

### 5.1 本地（Codex）
在 c:\GRA-CNN 执行：

powershell -NoProfile -ExecutionPolicy Bypass -Command "
$log='c:\GRA-CNN\experiments\stage_mid_pruning\results_v5_local_resnet20\runner_stdout.log';
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null;
$cmd=\"Set-Location 'c:\GRA-CNN'; & '.\\.venv\\Scripts\\python.exe' 'experiments\\stage_mid_pruning\\run_p1_validation_v5.py' --architectures resnet20 --methods L1,FPGM,GRA-v4,GRA-v5,Random --ratios 0.3,0.5,0.7 --seeds 42,123,456,789,1024 --num_workers 1 --timeout_sec 5400 --finetune_epochs 40 --result_dir experiments\\stage_mid_pruning\\results_v5_local_resnet20 *>> '$log'\";
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd -WindowStyle Hidden
"

### 5.2 服务器（Claude）
连接后执行：

ssh FatMachine
cd /d C:\Users\sshuser\GRA-CNN\project

powershell -NoProfile -ExecutionPolicy Bypass -Command "
$log='C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\runner_stdout.log';
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null;
$cmd=\"cd /d C:\Users\sshuser\GRA-CNN\project && C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe experiments\\stage_mid_pruning\\run_p1_validation_v5.py --architectures resnet56 --methods L1,FPGM,GRA-v4,GRA-v5,Random --ratios 0.3,0.5,0.7 --seeds 42,123,456,789,1024 --num_workers 1 --timeout_sec 5400 --finetune_epochs 40 --result_dir experiments\\stage_mid_pruning\\results_v5_server_resnet56 >> $log 2>&1\";
Start-Process cmd.exe -ArgumentList '/c',$cmd -WindowStyle Hidden
"

## 6. 监控命令（每次都给）

### 6.1 本地
- 进度计数:
  - powershell -Command "(Get-Content 'c:\GRA-CNN\experiments\stage_mid_pruning\results_v5_local_resnet20\checkpoint.json' | ConvertFrom-Json).completed.Count"
- 日志尾部:
  - Get-Content c:\GRA-CNN\experiments\stage_mid_pruning\results_v5_local_resnet20\run_log.txt -Tail 30
- GPU:
  - nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

### 6.2 服务器
- 进度计数:
  - ssh FatMachine "powershell -Command \"(Get-Content 'C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\checkpoint.json' | ConvertFrom-Json).completed.Count\""
- 日志尾部:
  - ssh FatMachine "powershell -Command \"Get-Content 'C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\run_log.txt' -Tail 30\""
- GPU:
  - ssh FatMachine "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"

## 7. 结果合并规范
- 本地结果:
  - c:\GRA-CNN\experiments\stage_mid_pruning\results_v5_local_resnet20\results.csv
- 服务器结果:
  - C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\results.csv
- 合并后统一字段:
  - architecture,dataset,method,ratio,iso_flops,seed,gra_version,baseline_acc,pruned_acc,final_acc,params_before,params_after,compression_ratio,pruning_scope,timestamp

## 8. 风险与处理
- 若出现 timeout:
  - 首先检查是否有并行重负载任务占用 GPU/CPU。
  - 保持 timeout_sec=5400，不要回退 1800。
- 若 checkpoint 不增长:
  - 看 run_log 是否有 RESULT_JSON。
  - 检查单任务脚本是否路径错误或环境缺包。
- 若服务器断线:
  - 使用后台启动方式，任务不会随终端关闭而停止。

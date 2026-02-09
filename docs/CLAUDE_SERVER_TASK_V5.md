# Claude 服务器任务单（GRA-v5）

## 任务目标
- 在服务器执行 v5 分流任务（resnet56 全矩阵 75 任务）。
- 保证断线不掉任务、日志完整、checkpoint 可恢复。

## 前置检查
1. ssh FatMachine
2. cd /d C:\Users\sshuser\GRA-CNN\project
3. git fetch origin
4. git checkout sync/v5-server-20260209
5. git pull --ff-only
6. 验证数据目录: dir C:\Users\sshuser\GRA-CNN\project\data\cifar-10-batches-py
7. 验证 checkpoint:
   - C:\Users\sshuser\GRA-CNN\project\checkpoints\resnet20_best.pth
   - C:\Users\sshuser\GRA-CNN\project\checkpoints\resnet56_best_new.pth

## 启动命令（后台）
powershell -NoProfile -ExecutionPolicy Bypass -Command "
$log='C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\runner_stdout.log';
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null;
$cmd=\"cd /d C:\Users\sshuser\GRA-CNN\project && C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe experiments\\stage_mid_pruning\\run_p1_validation_v5.py --architectures resnet56 --methods L1,FPGM,GRA-v4,GRA-v5,Random --ratios 0.3,0.5,0.7 --seeds 42,123,456,789,1024 --num_workers 1 --timeout_sec 5400 --finetune_epochs 40 --result_dir experiments\\stage_mid_pruning\\results_v5_server_resnet56 >> $log 2>&1\";
Start-Process cmd.exe -ArgumentList '/c',$cmd -WindowStyle Hidden
"

## 监控命令
- 进度:
  - powershell -Command "(Get-Content 'C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\checkpoint.json' | ConvertFrom-Json).completed.Count"
- 日志:
  - powershell -Command "Get-Content 'C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\run_log.txt' -Tail 40"
- GPU:
  - nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
- Python 进程:
  - powershell -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ? { $_.CommandLine -like '*run_p1_validation_v5.py*' -or $_.CommandLine -like '*single_task_v5.py*' } | select ProcessId,CommandLine"

## 验收标准
- checkpoint.json 可持续增长，最终 completed=75。
- results.csv 行数=75（不含表头）。
- run_log 无持续 timeout；若出现 timeout，记录发生时的 GPU/CPU 状态。

## 交付物
- C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\results.csv
- C:\Users\sshuser\GRA-CNN\project\experiments\stage_mid_pruning\results_v5_server_resnet56\run_log.txt
- 简短复盘:
  - 实际耗时
  - 成功/失败任务数
  - 异常与处理

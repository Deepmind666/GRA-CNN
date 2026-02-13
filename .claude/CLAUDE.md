# GRA-CNN 项目规则

## Environment
- Windows 11, PowerShell only, no bash-only syntax
- Python 3.11, conda env: gra311
- GPU: RTX 5090 D (24GB VRAM)
- 仅用 PowerShell 兼容命令；复杂命令优先脚本化（.ps1）
- 同类命令失败两次必须先诊断根因再继续
- 内存受限：不启动超过 2-3 个并发 Python 进程，不使用无界并行
- LaTeX in Markdown：反斜杠必须正确转义（\\frac 而非 \frac），避免 Python 字符串解释

## Context Management
- 积极节省上下文窗口：先用 grep/glob 定位，再读目标文件，不预读整个代码库
- 审计/审查时只读范围内的文件，不做广泛探索
- 大任务拆分为多阶段，每阶段结束后写中间结果到文件
- 用 Task sub-agent 做代码探索，保持主对话上下文干净

## Code Editing Rules
- 不直接修改已有正常工作的代码文件，除非明确要求
- 审查/grep 限定到明确文件或目录，禁止 broad glob 扫到归档/备份文件
- 结论必须可追溯到具体路径和行号；证据不足时标注"证据不足"
- 核心算法变更需用户明确批准
- 写 Python 文件时一次写完整，不做碎片化多步编辑

## Experiment Workflow
- 批量实验前必须先跑 smoke task，确认入口、输出、checkpoint 正常
- 长任务必须输出进度日志：[done/total]、配置、最新耗时、ETA
- 启动时必须打印 estimated_total_time, avg_task_time_sec, eta_basis
- 宣布完成前做 all-clear 复核：输出文件非空 + checkpoint 前进 + 二次 grep
- 不主动轮询实验进度，除非用户要求
- 顺序执行，不并行启动多个 GPU 任务
- 实验失败时先检查内存再重试

## Session Management
- 单轮默认最多推进 1 个主目标 + 1 个次目标
- 执行前先写：目标文件、禁触文件、完成标准；结束后写阶段快照
- 无增量不重复汇报；每次进展必须给 delta
- 每次 ETA 必须给计算依据（最近任务均值/日志时间戳差分）

## Language
- 所有回复使用中文
- 论文中方法名统一用 GRA（不带版本号）
- 用户最新要求优先级最高

## Quality Checks
- 宣布"已完成"前必须做验证 pass（grep/文件检查）
- 残留问题（forbidden assertions, stale metadata）必须在验证中捕获
- 审计声称"全部清除"前，必须重新 grep 验证零残留

## Project Structure
- 核心算法：pruning/core_algorithm_v7.py（当前版本）
- 实验入口：experiments/unified_worker.py
- 模型定义：models/resnet_cifar.py, models/vgg_cifar.py
- 结果目录：experiments/unified_results/
- Checkpoint：checkpoints/（ResNet）, experiments/（VGG baseline）
- 支持架构：resnet20, resnet56, vgg16
- 支持数据集：cifar10, cifar100

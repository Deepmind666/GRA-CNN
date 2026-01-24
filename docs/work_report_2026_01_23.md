# 工作纪录 - 2026年1月23日

## 1. 今日工作概览
今日主要完成了 GRA-CNN 算法的健壮性升级、RTX 5090 Blackwell 架构的环境适配，并启动了覆盖所有图表的并行实验矩阵。

## 2. 算法思路与创新点 (GRA-Fisher v2.0)
本项目的核心创新在于 **Global Relevant Analysis (GRA)** 与 **Fisher 信息量** 的深度融合，旨在解决传统剪枝方法在轻量化架构中容易出现的“语义丢失”问题。

- **GRA 语义分析**：通过计算神经元激活状态与模型 `logit margin` 的相关性，识别出对类别区分度起关键作用的“语义滤波器”。
- **MORF 指标 (Multi-Objective Relevant Fusion)**：
    - **Fisher (40%)**: 衡量一阶梯度敏感度。
    - **Orthogonality (20%)**: 引入通道直交性，减少特征冗余。
    - **GRA (25%)**: 捕获全局语义关联性。
    - **L1 Norm (15%)**: 维持基本的参数量级衡量。
- **创新点**：
    1. **跨架构支持**：成功将方法论从 ResNet 扩展到 **MobileNetV2**（带 Inverted Residual 结构）。
    2. **局部-全局协同**：Fisher 负责局部稳定性，GRA 负责全局语义，解决了剪枝过程中的“近视”问题。

## 3. 重要文件路径
- **论文稿件**：`[manuscript_apin.tex](file:///c:/GRA-CNN/APIN_Submission/manuscript_apin.tex)`
- **核心算法脚本**：`[run_real_pruning.py](file:///c:/GRA-CNN/experiments/run_real_pruning.py)`
- **实验调度器**：`[master_experiment_runner.py](file:///c:/GRA-CNN/experiments/master_experiment_runner.py)`
- **实验日志**：`[master_run_log.txt](file:///C:/GRA-CNN/experiments/master_run_log.txt)`
- **汇总数据**：`[supplementary_results.csv](file:///C:/GRA-CNN/experiments/supplementary_results.csv)`

## 4. 当前状态与下一步计划
- **状态**：正在进行 12 小时高强度并行实验。
- **下一步**：
    - 采集 ResNet-110 (CIFAR-100) 的深度剪枝数据。
    - 验证 Tiny-ImageNet 上的泛化能力。
    - 自动绘制 L1 vs GRA 的性能对比线图。

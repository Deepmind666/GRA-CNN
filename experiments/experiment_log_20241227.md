# GRA-CNN 实验日志 - 2024年12月27日

> **目标**: 完成APIN投稿级别的论文，包含完整的先验实验支撑和理论基础

---

## 📋 今日主要工作

### 1. 先验实验设计与实施 ✅

**目的**: 为GRA指标选择提供理论和实验支撑

| 实验 | 脚本 | 状态 | 输出 |
|------|------|------|------|
| A: L1 vs GRA 正交性 | `prior_exp_semantic_vs_l1.py` | ✅ 完成 | `fig_l1_vs_gra_scatter.pdf` |
| C: 层级分布分析 | `prior_exp_layerwise.py` | ✅ 完成 | `fig_layerwise_analysis.pdf` |
| D: 类别特异性 | `prior_exp_classwise.py` | ✅ 完成 | `fig_classwise_importance.pdf` |
| E: 信息论/熵分析 | `prior_exp_entropy.py` | ✅ 完成 | `fig_entropy_analysis.pdf` |

**关键发现**:
- Pearson r = 0.0146 (L1与GRA近乎零相关)
- 28% 通道为"隐藏语义通道" (低L1, 高GRA)
- 22% 通道为"假阳性" (高L1, 低GRA)

---

### 2. 综合实验矩阵

**配置**:
- 架构: ResNet-20, ResNet-56, ResNet-110, VGG-16
- 数据集: CIFAR-10, CIFAR-100
- 方法: L1, FPGM, HRank, GRA
- 剪枝率: 0.3, 0.5, 0.7
- 种子: 0, 1, 2

**运行命令**:
```powershell
# 主综合实验 (72 runs)
C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe run_comprehensive.py --archs resnet56 resnet20 --datasets cifar10 cifar100 --methods l1 fpgm hrank gra --ratios 0.3 0.5 0.7 --seeds 0 1 2 --save-dir C:\GRA-CNN\experiments\comprehensive

# ResNet-110 实验
C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe run_comprehensive.py --archs resnet110 --datasets cifar10 --methods l1 gra --ratios 0.3 0.5 0.7 --seeds 0 1 2 --save-dir C:\GRA-CNN\experiments\comprehensive
```

---

### 3. 核心创新点

1. **语义关联剪枝范式**: 首次将GRA用于通道重要性评估，捕获与决策边界的功能耦合
2. **无梯度、尺度不变**: 不需要反向传播，对跨层激活差异鲁棒
3. **正交于现有方法**: 与L1捕获完全不同的信息维度

---

## 🔧 关键路径与配置

### 环境
```
Python: C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe
GPU: NVIDIA RTX 5090 (高性能)
```

### 项目结构
```
C:\GRA-CNN\
├── models/
│   ├── resnet_cifar.py      # ResNet-20/56/110
│   ├── vgg_cifar.py         # VGG-16
│   └── mobilenetv2.py       # MobileNetV2 (新增)
├── pruning/
│   ├── gra_score.py         # GRA 核心评分
│   ├── l1_score.py
│   ├── fpgm_score.py
│   ├── hrank_score.py
│   └── extra_scorers.py     # Taylor, ACP (新增)
├── experiments/
│   ├── run_comprehensive.py # 综合实验运行器
│   ├── prior_exp_*.py       # 先验实验脚本
│   └── prior_experiments/   # 先验实验输出
└── APIN_Submission/
    ├── manuscript_apin.tex  # 论文源文件
    ├── vis/                 # 可视化脚本
    └── *.pdf                # 图表
```

### 生成的图表
```
C:\GRA-CNN\APIN_Submission\
├── fig_l1_vs_gra_scatter.pdf     # L1 vs GRA 散点图
├── fig_layerwise_analysis.pdf    # 层级分析
├── fig_classwise_importance.pdf  # 类别特异性热力图
├── fig_entropy_analysis.pdf      # 信息论分析
├── fig2_pro.pdf                  # ResNet-20 专业图
├── fig3_pro.pdf                  # ResNet-56 专业图
├── fig_cifar100_bar_pro.pdf      # CIFAR-100 柱状图
├── fig_tiny_pro.pdf              # Tiny-ImageNet
└── fig_rho_pro.pdf               # ρ敏感性
```

---

## 📝 待完成任务

- [ ] 综合实验完成后生成12面板网格图
- [ ] 将先验实验图表整合到论文
- [ ] 添加理论章节 (信息论解释)
- [ ] 运行Tiny-ImageNet实验
- [ ] 最终PDF编译

---

## ⏱️ 时间线

| 时间 | 事件 |
|------|------|
| 12:35 | 开始图表专业化升级 |
| 12:53 | 用户反馈图表不够丰富，开始扩充实验 |
| 13:08 | 开始运行综合实验 (72 runs) |
| 13:25 | 设计理论基础和先验实验 |
| 13:40 | 先验实验完成，生成4个理论支撑图表 |
| 14:05 | 服务器重启，记录日志并恢复实验 |
| 14:06 | 启动3个并行实验: 72-run主矩阵 + ResNet-110 + VGG-16 |

---

## 🔄 当前运行的实验进程

| 进程ID | 实验内容 | 状态 |
|--------|---------|------|
| f067ee1b | ResNet-20/56 × CIFAR-10/100 × 4方法 × 3剪枝率 × 3seeds (72 runs) | 🔄 运行中 |
| d557c75f | ResNet-110 × CIFAR-10 × L1+GRA × 3剪枝率 × 3seeds (18 runs) | 🔄 运行中 |
| 904c426b | VGG-16 × CIFAR-10 × L1+GRA × 3剪枝率 × 3seeds (18 runs) | 🔄 运行中 |

---

**状态**: 🔄 3个并行实验运行中

---

# 12月28日 (续)

## 📊 进度总结 (截至 00:35)

### 已完成的实验
| 实验 | 状态 | 结果 |
|------|------|------|
| **先验实验 (4个)** | ✅ 完成 | 4个理论支撑图表 |
| **ResNet-110 CIFAR-10** | ✅ 完成 | 18 runs, GRA 87.42% vs L1 86.90% (0.7 ratio) |
| **VGG-16 CIFAR-10 (L1)** | ✅ 完成 | 9 runs, ~91.75% |
| **VGG-16 CIFAR-10 (GRA)** | 🔶 部分 | 2/9 runs, GRA 92.1% vs L1 91.74% |

### 待完成的实验
| 实验 | 规模 | 预计时间 |
|------|------|---------|
| **主矩阵 (ResNet-20/56 × CIFAR-10/100)** | 72 runs | ~2-3小时 |
| **VGG-16 GRA 剩余** | 7 runs | ~30分钟 |

### 关键发现 (来自已完成实验)
- **ResNet-110**: GRA 在高剪枝率 (0.7) 表现更佳, +0.52% vs L1
- **VGG-16**: GRA 初步显示 +0.36% 优势
- **正交性**: Pearson r = 0.0146 证明 GRA 捕获独立信息

## 🔄 当前恢复的实验

| 进程 | 内容 | 状态 |
|------|------|------|
| 主矩阵 | ResNet-20/56 × CIFAR-10/100 × 4方法 × 3seeds | 🔄 启动中 |
| VGG-16 | GRA × 3剪枝率 × 3seeds | 🔄 启动中 |

**预计完成时间**: ~3小时 (约 03:30)

---

## 🚀 10小时实验扩展 (00:38)

用户批准10小时实验预算，大规模扩展实验矩阵：

### 新增实验配置
| 维度 | 原配置 | 扩展后 |
|------|--------|--------|
| 方法 | 4种 (L1, FPGM, HRank, GRA) | 6种 (+Taylor, ACP) |
| 剪枝率 | 3种 (0.3, 0.5, 0.7) | 7种 (0.2-0.8) |
| 架构 | 4种 | 保持 |
| 总实验数 | ~100 runs | ~400+ runs |

### 当前并行运行的进程

| 进程ID | 内容 | 规模 | 状态 |
|--------|------|------|------|
| 81f13eba | ResNet-56/20 CIFAR-10/100 × 4方法 | 72 runs | 🔄 |
| a44b443d | VGG-16 GRA | 9 runs | 🔄 |
| dd9e1448 | ResNet-56 CIFAR-10 × 6方法 × 7剪枝率 | 126 runs | 🔄 |
| 44f4366b | ResNet-20 CIFAR-10 × 6方法 × 7剪枝率 | 126 runs | 🔄 |
| 2d000910 | ResNet-56 CIFAR-100 × 4方法 | 36 runs | 🔄 |
| fa75dd3f | ResNet-20 CIFAR-100 × 4方法 | 36 runs | 🔄 |
| bf8c6d9a | 收敛分析实验 | 2方法 × 40 epochs | 🔄 |

### 新增实验脚本
- `exp_convergence.py` - 收敛曲线对比
- `exp_statistical_tests.py` - 统计显著性检验 (t-test, Cohen's d)

**总计**: ~400+ 实验并行运行
**预计完成**: 10小时内 (~10:30)

---

## 📋 12月28日 11:05 更新

### 10小时实验结果汇总

**已完成**:
- ✅ VGG-16 CIFAR-10: L1 + GRA 各9 runs (18 total)
- ✅ ResNet-110 CIFAR-10: L1 + GRA 各9 runs (18 total)
- ✅ 收敛分析: L1 vs GRA × 40 epochs
- ✅ 先验实验: 5个图表

**部分完成/中断**:
- ❌ 主72-run矩阵: 仅7 runs (内存问题中断)
- ❌ 扩展实验: 未完成

### 🔄 顺序实验队列启动

创建 `run_queue.py` 避免GPU内存溢出，顺序运行:

| 序号 | 实验 | 规模 | 状态 |
|------|------|------|------|
| 1 | ResNet-56 CIFAR-10 × 4方法 | 36 runs | 🔄 |
| 2 | ResNet-20 CIFAR-10 × 4方法 | 36 runs | ⏳ |
| 3 | ResNet-56 CIFAR-100 × 4方法 | 36 runs | ⏳ |
| 4 | ResNet-20 CIFAR-100 × 4方法 | 36 runs | ⏳ |

**进程ID**: ab84eed6
**预计时间**: ~3-4小时

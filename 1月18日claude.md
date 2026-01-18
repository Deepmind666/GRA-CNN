# GRA-CNN 项目完整文档
**最后更新**: 2026年1月18日  
**维护者**: Claude (AI Assistant)

---

## 1. 项目概述

### 1.1 研究目标
GRA-CNN 是一种基于**灰色关联分析 (Gray Relational Analysis)** 的 CNN 结构化剪枝方法，目标是发表在 **Applied Intelligence (APIN)** 期刊。

### 1.2 核心创新点
传统剪枝方法 (L1-Norm, FPGM, HRank) 基于**结构特征**（权重大小、几何冗余），而 GRA-CNN 基于**语义对齐**：
- 衡量每个通道的激活模式与最终分类 logit 的同步程度
- 保留"语义滤波器"（与决策高度相关的通道）
- 剔除"结构冗余"（能量大但语义无关的通道）

### 1.3 关键公式
```
γ_c = (1/N) Σ ξ_c(i)                    # 通道 c 的 GRA 分数
ξ_c(i) = (Δ_min + ρ·Δ_max) / (|a_c(i) - z(i)| + ρ·Δ_max)
```
- `a_c(i)`: 通道 c 在样本 i 上的激活值
- `z(i)`: 样本 i 的输出 logit
- `ρ = 0.5`: 分辨率系数 (默认值)

---

## 2. 项目文件结构

```
C:\GRA-CNN\
│
├── 📄 评审意见文档
│   ├── 1月11日gemini评审意见.md    # Gemini 评审 (创新/图表/实验)
│   ├── 1月18日gpt评审.md           # GPT 评审 (方法改进/叙事)
│   └── 1月18日claude.md            # 本文档
│
├── 📁 APIN_Submission/             # ⭐ 论文投稿目录
│   ├── manuscript_apin.tex         # 主论文 LaTeX 源文件
│   ├── manuscript_apin.pdf         # 编译后 PDF (23页)
│   ├── manuscript_final.bib        # 参考文献
│   ├── sn-jnl.cls                  # Springer Nature 模板
│   │
│   ├── 📊 主要图表
│   │   ├── fig2_nature_12panel.pdf    # Figure 2: 12面板精度曲线
│   │   ├── fig3_tiny_composite.pdf    # Figure 3: Tiny-ImageNet 分析
│   │   ├── fig4_convergence_pro.pdf   # Figure 4: 收敛曲线
│   │   ├── fig5_sensitivity_pro.pdf   # Figure 5: Rho 敏感性
│   │   ├── fig_orthogonality.pdf      # GRA vs L1 正交性分析
│   │   └── fig_pareto.pdf             # 帕累托前沿图
│   │
│   └── 📁 vis/                     # 可视化脚本
│       ├── generate_fig3_HONEST.py    # ⭐ 诚实版 Fig3 (使用真实数据)
│       ├── create_scientific_figures_fixed.py  # Fig2 生成器
│       ├── table_ablation.tex         # 消融实验表格
│       └── table_performance_pro.tex  # 性能表格
│
├── 📁 experiments/                 # ⭐ 实验结果 (核心目录)
│   │
│   ├── 📊 数据汇总文件
│   │   ├── supplementary_results.csv      # 主结果 (547条CIFAR记录)
│   │   ├── REAL_tiny_imagenet_data.csv    # 真实Tiny-ImageNet数据
│   │   ├── comprehensive_10hr_results.csv # 10小时通宵实验结果
│   │   └── overnight_progress.json        # 通宵实验进度
│   │
│   ├── 📁 基线模型 (已训练)
│   │   ├── baseline_cifar10_resnet20.pth  # ~1.1MB
│   │   ├── baseline_cifar10_resnet56.pth  # ~3.5MB
│   │   ├── baseline_cifar10_vgg16.pth     # ~59MB
│   │   ├── baseline_cifar100_*.pth        # CIFAR-100 基线
│   │   └── baseline_tiny_resnet18_60ep.pth # Tiny-ImageNet 基线
│   │
│   ├── 📁 实验子目录 (命名规范: {dataset}_{arch}_{method}_{ratio})
│   │   ├── cifar10_resnet20_gra_0.5/      # 包含训练日志和模型
│   │   ├── cifar10_resnet56_l1_0.3/
│   │   ├── cifar100_resnet20_fpgm_0.5/
│   │   └── ... (共 94+ 个实验目录)
│   │
│   ├── 📁 tinyimagenet/            # Tiny-ImageNet 实验
│   │   ├── tinyimagenet_gra_0.3_0.5.csv
│   │   ├── tinyimagenet_gra_0.5_0.5.csv
│   │   ├── tinyimagenet_l1_0.3_0.5.csv
│   │   └── ...
│   │
│   └── 🔧 核心脚本
│       ├── run_real_pruning.py        # ⭐ 主剪枝实验脚本
│       ├── overnight_experiment_suite.py  # 通宵批量实验
│       ├── harvest_results.py         # 结果收集器
│       ├── measure_efficiency.py      # FLOPs/吞吐量测量
│       └── train_baseline.py          # 基线模型训练
│
├── 📁 models/                      # 网络架构定义
│   ├── resnet_cifar.py             # ResNet-20/32/44/56/110 for CIFAR
│   ├── resnet18_tiny.py            # ResNet-18 for Tiny-ImageNet
│   ├── vgg_cifar.py                # VGG-16 for CIFAR
│   └── mobilenetv2.py              # MobileNetV2 (备用)
│
├── 📁 pruning/                     # 剪枝算法实现
│   ├── gra_score.py                # ⭐ GRA 重要性评分
│   ├── l1_score.py                 # L1-Norm 评分
│   ├── fpgm_score.py               # FPGM 评分
│   ├── hrank_score.py              # HRank 评分
│   ├── extra_scorers.py            # 统一评分接口
│   └── prune_model.py              # 模型剪枝执行器
│
├── 📁 data/                        # 数据集 (自动下载)
│   ├── cifar-10-batches-py/
│   ├── cifar-100-python/
│   └── tiny-imagenet-200/          # 需手动下载
│
└── 📁 archive/                     # 历史版本/备份
```

---

## 3. 实验配置矩阵

### 3.1 已完成实验 (547+ 条记录)

| 架构 | 数据集 | 方法 | 剪枝率 | 状态 |
|------|--------|------|--------|------|
| ResNet-20 | CIFAR-10 | GRA/L1/FPGM/HRank | 0.3~0.7 | ✅ |
| ResNet-32 | CIFAR-10 | GRA/L1/FPGM/HRank | 0.3~0.7 | ✅ |
| ResNet-44 | CIFAR-10 | GRA/L1/FPGM/HRank | 0.3~0.7 | ✅ |
| ResNet-56 | CIFAR-10 | GRA/L1/FPGM/HRank | 0.3~0.7 | ✅ |
| ResNet-110 | CIFAR-10 | GRA/L1/FPGM/HRank | 0.3~0.7 | ✅ |
| VGG-16 | CIFAR-10 | GRA/L1/FPGM/HRank | 0.3~0.7 | ✅ |
| ResNet-20~110 | CIFAR-100 | GRA/L1/FPGM/HRank | 0.3~0.7 | ✅ |
| VGG-16 | CIFAR-100 | GRA/L1/FPGM/HRank | 0.3~0.7 | ✅ |
| ResNet-18 | Tiny-ImageNet | GRA/L1/FPGM | 0.3,0.5,0.7 | ✅ |
| ResNet-18 | Tiny-ImageNet | HRank | - | ❌ 缺失 |

### 3.2 基线精度参考

| 架构 | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|------|----------|-----------|---------------|
| ResNet-20 | 91.70% | 67.50% | - |
| ResNet-56 | 93.20% | 71.50% | - |
| ResNet-110 | 93.80% | 74.30% | - |
| VGG-16 | 93.50% | 73.00% | - |
| ResNet-18 | - | - | 70.50% |

---

## 4. 快速开始

### 4.1 环境配置
```bash
# Python 环境
C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe

# 主要依赖
pip install torch torchvision pandas numpy matplotlib scipy thop
```

### 4.2 运行单个剪枝实验
```bash
cd C:\GRA-CNN
python experiments/run_real_pruning.py \
    --arch ResNet-56 \
    --dataset CIFAR-10 \
    --method GRA \
    --ratio 0.5 \
    --epochs 40
```

### 4.3 批量运行实验
```bash
python experiments/overnight_experiment_suite.py  # 运行82个实验
```

### 4.4 生成论文图表
```bash
# Figure 2: 12面板精度曲线
python APIN_Submission/vis/create_scientific_figures_fixed.py

# Figure 3: Tiny-ImageNet (诚实版)
python APIN_Submission/vis/generate_fig3_HONEST.py
```

### 4.5 编译论文 PDF
```bash
cd C:\GRA-CNN\APIN_Submission
"C:\Users\admin\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode manuscript_apin.tex
```

### 4.6 测量模型效率
```bash
python experiments/measure_efficiency.py  # 测量 FLOPs 和吞吐量
```

---

## 5. 科研严谨性审计 (2026-01-18)

### 5.1 发现的问题及修复状态

| 问题 | 严重程度 | 状态 | 说明 |
|------|----------|------|------|
| Figure 3 精度硬编码 | 🔴 严重 | ✅ 已修复 | 改用 experiments/tinyimagenet/*.csv 真实数据 |
| FLOPs 估算不准 | 🟡 中等 | ✅ 已改进 | 用 thop 测量基线: 2.23 GFLOPs |
| HRank 无 Tiny 实验 | 🟡 中等 | ✅ 已处理 | 从 Figure 3 中移除 |
| 吞吐量未实测 | 🟡 中等 | ⏸️ 待做 | 需 GPU 基准测试 |

### 5.2 真实 vs 硬编码数据对比

| 方法 | 剪枝率 | 真实精度 | 硬编码精度 | 误差 |
|------|--------|----------|------------|------|
| GRA | 0.5 | **66.71%** | 63.1% | +3.61% |
| L1 | 0.5 | **64.29%** | 58.9% | +5.39% |
| FPGM | 0.5 | **62.58%** | 59.5% | +3.08% |

### 5.3 数据溯源

- **CIFAR 数据**: `experiments/supplementary_results.csv` (547条, 有时间戳)
- **Tiny-ImageNet 数据**: `experiments/tinyimagenet/*.csv` (12条真实记录)
- **基线 FLOPs**: 2.23 GFLOPs (thop 测量 ResNet-18)

---

## 6. 论文结构概览

| 章节 | 内容 | 对应图表 |
|------|------|----------|
| Abstract | 核心贡献 (+2.42% on Tiny-ImageNet) | - |
| Introduction | 问题定义, GRA 动机 | Fig 1 (框架图) |
| Related Work | 剪枝方法综述 | - |
| Method | GRA 算法, 公式推导 | Algorithm 1 |
| Experiments | 12配置对比, 消融实验 | Fig 2, 3, Table 1-3 |
| Analysis | 正交性, 收敛性, Rho敏感性 | Fig 4, 5, 6 |
| Discussion | 语义滤波器概念, 局限性 | Table 4 |
| Conclusion | 总结与展望 | - |

---

## 7. 待办事项

### 高优先级
- [ ] 补充 HRank Tiny-ImageNet 实验
- [ ] 实测剪枝模型的精确 FLOPs
- [ ] GPU 吞吐量基准测试

### 中优先级  
- [ ] 实现可学习的 GRA (注意力机制版本)
- [ ] 自适应逐层剪枝率
- [ ] ImageNet 大规模验证

### 低优先级
- [ ] ViT/ConvNeXt 架构扩展
- [ ] 目标检测/语义分割任务验证

---

## 8. 常见问题

### Q: 如何添加新的剪枝方法?
在 `pruning/extra_scorers.py` 中实现 `score_channels(model, dataloader)` 函数。

### Q: 数据集存放位置?
默认 `C:\GRA-CNN\data\`，代码会自动下载 CIFAR。Tiny-ImageNet 需手动下载到 `data/tiny-imagenet-200/`。

### Q: 实验结果存放在哪?
每次实验创建 `experiments/{dataset}_{arch}_{method}_{ratio}/` 目录，包含:
- `train_log.csv`: 训练曲线
- `model_pruned.pth`: 剪枝后模型
- `result.json`: 最终精度

---

*文档结束 - 如有问题请联系项目维护者*

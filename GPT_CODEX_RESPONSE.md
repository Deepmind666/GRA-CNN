# 对 GPT Codex 审查意见的回复

## 审查意见原文摘要

GPT Codex 指出5个问题：
1. 版本号与文档不一致 (代码是5.2，摘要说5.3)
2. BN冻结修复未落地 (model.train()会重新切回训练态)
3. ResNet FLOPs修复不完整 (双重下采样)
4. 实验结果不支持"比L1/FPGM/HRank高10+点"
5. 自审文档仍是v5.0/v5.2

---

## 逐条回复

### 问题1：版本号不一致

**Codex意见**: 代码是5.2，摘要说5.3

**回复**: ⚠️ **版本已更新，Codex判断对旧版本成立**

**时间线澄清**:
- Codex审查时基于旧版本，当时确实是5.2
- 后续已更新为5.3，当前代码第25行: `GRA_VERSION = "5.3"`

**证据**: `test_version()` 通过，断言 `GRA_VERSION == "5.3"`

---

### 问题2：BN冻结修复未落地

**Codex意见**: 代码是"先BN.eval()，再model.train()"，model.train()会把BN重新切回训练态。

**回复**: ⚠️ **核心路径已实现，但测试覆盖不完整**

**已验证**:
- `collect_activations_margins_and_gradnorms()` 使用 `model.eval()` + finally恢复
- 单元测试 `test_bn_freezing()` 验证 running_mean/var 不变

**待补充**:
- `compute_ngscs_per_layer()` 异常场景未测试
- `estimate_hessian_diagonal_hutchinson()` 异常场景未测试
- `compute_fisher_information()` 异常场景未测试
- Dropout层影响未验证

---

### 问题3：ResNet FLOPs修复不完整

**Codex意见**: stride=2 conv使用的stage_sizes已是下采样后尺寸，estimate_layer_flops又除一次stride，双重下采样。

**回复**: ⚠️ **CIFAR-ResNet已修复，ImageNet-ResNet仍不完整**

**已修复 (CIFAR-ResNet)**:
- `stage_input_sizes` 存储INPUT尺寸
- layer2 stride=2 conv: 输入32 → 输出16

**待修复 (ImageNet-ResNet)**:
- stem层7x7 conv + maxpool未建模
- 输入尺寸224的池化路径未处理

**测试不足**: `test_resnet_downsample_flops()` 只打印不断言

---

### 问题4：实验结果不支持"高10+点"

**Codex意见**: v53_results只含GRA，无L1/FPGM/HRank，无法证明优于基线。

**回复**: ✅ **Codex完全正确，"高10+点"结论不成立**

**问题**:
- v53_results 只有GRA结果，无同条件基线
- final_consolidated 中的L1/FPGM/HRank是**非同条件历史数据**
- 不能用于支撑强结论

**必须补充**: 用v5.3代码同条件跑L1/FPGM/HRank

---

### 问题5：自审文档仍是v5.0/v5.2

**Codex意见**: 没有v5.3的对应自审文档。

**回复**: ✅ **Codex指出正确**

确实缺少 `SELF_REVIEW_V53.md`，需要补充。

---

## 总结

| 问题 | Codex判断 | 修订后结论 |
|------|-----------|------------|
| 1. 版本号 | 5.2 | ⚠️ 已更新为5.3，Codex对旧版成立 |
| 2. BN冻结 | 未落地 | ⚠️ 核心路径已实现，测试覆盖不完整 |
| 3. FLOPs | 双重下采样 | ⚠️ CIFAR已修复，ImageNet待修复 |
| 4. 基线对比 | 缺失 | ✅ 正确，必须补充同条件实验 |
| 5. 自审文档 | 缺失 | ✅ 正确，需创建v5.3自审 |

---

## 后续行动计划

### 必须完成

1. **同条件对比实验**: 用v5.3代码跑L1/FPGM/HRank (优先级最高)
2. **v5.3自审文档**: 创建 `SELF_REVIEW_V53.md`
3. **补充单元测试**: 覆盖其他梯度函数的BN冻结

### 建议完成

4. **ImageNet FLOPs**: 修复stem/pool路径
5. **多次运行**: 3-5次取平均，增强统计显著性
6. **测试断言**: 将打印改为断言


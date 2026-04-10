---
title: "MoBiE: MoE二值化专家"
date: 2026-04-08
type: paper
tags:
  - 模型压缩
  - MoE
  - 量化
---

# MoBiE: MoE二值化专家

**arxiv:** [2604.06798](https://arxiv.org/abs/2604.06798)
**作者:** Zhixiong Zhao et al.

### 核心问题

MoE 模型的高内存和计算成本使其部署困难，但现有二值化方法针对密集 LLM 设计，难以处理 MoE 的三个特有挑战：

1. **跨专家冗余** (Cross-expert redundancy)
2. **任务无关的重要性估计** (Task-agnostic importance estimation)
3. **量化导致的路由偏移** (Quantization-induced routing shifts)

### 核心贡献

| 创新 | 解决的问题 | 方法 |
|------|-----------|------|
| CEJD | 跨专家冗余 | 联合 SVD 分解 |
| GLAS | 重要性估计 | 全局 loss 梯度 |
| NGES | 路由偏移 | 零空间投影 |

### 架构

```
┌─────────────────────────────────────────────────────────┐
│                    MoBiE 架构                            │
├─────────────────────────────────────────────────────────┤
│  §4.1 CEJD: Cross-Expert Joint Decomposition           │
│  - 专家权重联合 SVD 分解                                │
│  - 减少跨专家冗余                                        │
├─────────────────────────────────────────────────────────┤
│  §4.2 GLAS: Global Loss-Aligned Saliency               │
│  - 将全局 loss 梯度融入 Hessian                          │
│  - 任务感知的重要性估计                                  │
├─────────────────────────────────────────────────────────┤
│  §4.3 NGES: Null-space Guided Expert-Shift Suppression │
│  - 输入零空间投影                                        │
│  - 减轻路由失真                                          │
└─────────────────────────────────────────────────────────┘
```

### 数学方法

**问题：二值化后专家偏移**
```
Binarization leads to expert-shift, where token assignments
migrate across experts compared to the original distribution
```

**GLAS 重要性估计：**
```
L_guided = || (∂ℓ/∂Z) ⊙ (Z - Ŷ) ||²_F
```

**NGES 零空间投影：**
```
R'(Router) = R + P_⊥ · Δ
其中 P_⊥ 是零空间投影矩阵
```

### 伪代码实现

```python
class MoBiE:
    def __init__(self, model):
        self.model = model
        self.experts = model.experts

    def cejd_decompose(self, rank=64):
        """
        Cross-Expert Joint Decomposition
        减少跨专家冗余
        """
        # 对所有专家权重进行联合 SVD
        W_all = torch.stack([e.weight for e in self.experts], dim=0)
        # W_all shape: [num_experts, in_features, out_features]

        # 对每个专家应用相同的低秩分解
        for i, expert in enumerate(self.experts):
            U_i, S_i, V_i = torch.svd_lowrank(W_all[i], n_components=rank)
            # 重构专家权重
            expert.weight.data = U_i @ torch.diag(S_i) @ V_i.T

    def glas_importance(self, calibration_data):
        """
        Global Loss-Aligned Saliency
        任务感知的重要性估计
        """
        # Step 1: 计算全局 loss 梯度
        gradients = self.compute_loss_gradients(calibration_data)

        # Step 2: 构建任务感知的 Hessian
        # H_task 包含下游 loss 的二阶信息
        H_task = torch.outer(gradients, gradients)

        # Step 3: 结合局部 Hessian 和全局信息
        # 局部 Hessian (GPTQ): sij = wij² * [H⁻¹]ii
        s_local = self.local_hessian

        # GLAS: 任务感知的重要性分数
        s_glas = s_local * (1 + torch.abs(H_task))
        return s_glas

    def nges_correct(self, router_output, input_features, alpha=0.1):
        """
        Null-space Guided Expert-Shift Suppression
        减轻路由偏移
        """
        # Step 1: 计算输入特征的零空间
        # 零空间是不影响输出的方向
        U, S, V = torch.svd_lowrank(input_features, n_components=rank)

        # Step 2: 构建零空间投影矩阵
        # P_⊥ = I - UU^T
        I = torch.eye(U.shape[0], device=U.device)
        P_perp = I - U @ U.T

        # Step 3: 计算路由偏移量
        # 偏移应该只在不影响输出的方向上
        delta = alpha * (P_perp @ router_output)

        # Step 4: 修正路由
        corrected = router_output + delta
        return F.softmax(corrected, dim=-1)

    def full_binarization_pipeline(self, calibration_data):
        """完整的 MoBiE 二值化流程"""
        # 1. CEJD: 减少跨专家冗余
        self.cejd_decompose(rank=64)

        # 2. GLAS: 校准重要性
        importance = self.glas_importance(calibration_data)

        # 3. 应用二值化权重
        for expert in self.experts:
            expert.weight.data = self.binarize_with_importance(
                expert.weight.data, importance
            )

        # 4. NGES: 修正路由
        router_output = self.model.router(input_features)
        self.model.router.weight.data = self.nges_correct(
            router_output, input_features
        )

    def binarize_with_importance(self, weight, importance):
        """基于重要性进行二值化"""
        # 只保留最重要的权重
        threshold = torch.quantile(torch.abs(weight), q=0.5)
        binary_weight = torch.sign(weight)
        binary_weight = binary_weight * (torch.abs(weight) > threshold).float()
        return binary_weight
```

### 实验结果

| 模型 | 指标 | 提升 |
|------|------|------|
| Qwen3-30B-A3B | 困惑度降低 | 52.2% |
| Qwen3-30B-A3B | 零样本性能 | +43.4% |
| Qwen3-30B-A3B | 推理加速 | 2× |

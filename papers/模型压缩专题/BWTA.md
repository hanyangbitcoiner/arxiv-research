---
title: "BWTA: 二值化Transformer"
date: 2026-04-05
type: paper
tags:
  - 模型压缩
  - Transformer
  - 二值化
---

# BWTA: 二值化Transformer

**arxiv:** [2604.03957](https://arxiv.org/abs/2604.03957)
**作者:** Yifu Ding et al.

### 核心问题

1. **精度损失**：超低比特量化导致精度大幅下降
2. **GPU 支持有限**：缺乏实用的 ultra-low bit CUDA kernel

### 核心贡献

1. **Binary Weights & Ternary Activations**：保持零点附近小值的分布
2. **Smooth Multi-Stage Quantization**：稳定快速收敛的训练方法
3. **BWTA MatMul CUDA Kernel**：指令级并行 bitpack

### 方法详解

**训练：Smooth Multi-Stage Quantization**
```
Smooth Multi-Stage Quantization:
┌─────────────────────────────────────────┐
│ Stage 1: FP16 → FP8                     │
│ Stage 2: FP8 → INT4                     │
│ Stage 3: INT4 → Ternary (BWTA)         │
└─────────────────────────────────────────┘

Levelwise Degradation Strategy:
- 保持零点附近值分布均匀
- 避免量化失真

Magnitude-Alignment Projection Factor:
- 缩放因子对齐
- 稳定训练曲线
```

**推理：BWTA MatMul Kernel**
```
Binary Weight & Ternary Activation MatMul
- 权重二值化：sign function
- 激活三值化：{-1, 0, +1}
- 位运算替代矩阵乘法
```

### 伪代码实现

```python
class BWTALinear(nn.Module):
    """Binary Weight & Ternary Activation Linear Layer"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def binarize_weight(self, weight):
        """
        权重二值化：符号函数
        将浮点权重映射到 {-1, +1}
        """
        return torch.sign(weight)

    def ternaryize_activation(self, x):
        """
        激活值三值化：
        保留零点附近小值的分布，映射到 {-1, 0, +1}

        关键洞察：
        - 零点附近的小值包含重要信息，不应丢弃
        - 三值化保持 sign(0) = 0 的特性
        """
        # 使用标准差作为阈值
        threshold = 0.5 * torch.std(x)

        # 三值化映射
        return torch.where(
            x > threshold,
            torch.ones_like(x),                          # 正值 -> +1
            torch.where(
                x < -threshold,
                -torch.ones_like(x),                    # 负值 -> -1
                torch.zeros_like(x)                     # 零点附近 -> 0
            )
        )

    def forward(self, x):
        # 三值化激活
        x_ternary = self.ternaryize_activation(x)
        # 二值化权重
        w_binary = self.binarize_weight(self.weight)
        # Bitwise 运算代替矩阵乘法
        return bitwise_matmul(x_ternary, w_binary)


class BWTAAttention(nn.Module):
    """BWTA 实现的 Attention"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.q_linear = BWTALinear(hidden_size, hidden_size)
        self.k_linear = BWTALinear(hidden_size, hidden_size)
        self.v_linear = BWTALinear(hidden_size, hidden_size)
        self.o_linear = BWTALinear(hidden_size, hidden_size)

    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # QK T 计算使用位运算
        # BWTA QK MatMul
        attn_scores = self.bwta_qk_matmul(Q, K)

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # OV 计算使用位运算
        # BWTA Att MatMul
        output = self.bwta_att_matmul(attn_probs, V)

        return self.o_linear(output)

    def bwta_qk_matmul(self, Q, K):
        """
        BWTA QK 实现
        使用 popcount 替代浮点乘法
        """
        # 打包成比特
        Q_packed = packbits(Q > 0, 'uint32')
        K_packed = packbits(K > 0, 'uint32')

        # XOR + popcount
        # (Q_i * K_i) ≈ popcount(~(Q_i XOR K_i)) for binary
        xored = Q_packed ^ K_packed
        matches = popcount(~xored)  # 匹配位数

        return matches.float()

    def bwta_att_matmul(self, attn_probs, V):
        """
        BWTA Attention MatMul 实现
        """
        # 解包概率
        probs_unpacked = unpackbits(attn_probs > 0.5, 'float32')

        # 乘法转加法
        # probs * V ≈ sum(probs_unpacked * V_bits)
        return torch.matmul(probs_unpacked, V)


class BWTAMatMul:
    """CUDA Kernel 实现"""

    @staticmethod
    def packbits_to_uint32(tensor):
        """
        将布尔/浮点tensor打包成uint32
        每32个比特位打包成一个uint32
        """
        # 简化版本：实际CUDA实现更复杂
        binary = (tensor > 0).int()
        packed = torch.zeros(
            binary.shape[0],
            (binary.shape[1] + 31) // 32,
            dtype=torch.uint32,
            device=tensor.device
        )
        for i in range(32):
            packed |= (binary[:, i::32].long() << i)
        return packed

    @staticmethod
    def popcount(tensor_uint32):
        """GPU上的popcount实现"""
        # 使用CUDA builtin
        return torch.zeros_like(tensor_uint32).float()

    @staticmethod
    def cuda_kernel_qk(x_packed, k_packed):
        """
        Attention QK 的 BWTA CUDA 实现

        优化点：
        1. 指令级并行：一次处理32个比特
        2. 寄存器优化：减少内存访问
        3. 异步拷贝：掩盖内存延迟
        """
        # XOR 操作
        xored = x_packed ^ k_packed
        # popcount: 统计1的个数
        matches = popcount(~xored)
        return matches


class SmoothMultiStageQuantization:
    """
    平滑多阶段量化训练框架
    渐进式将权重从 FP16 量化到 Ternary
    """

    def __init__(self, model, stages=[16, 8, 4, 3]):
        self.model = model
        self.stages = stages  # [FP16, FP8, INT4, Ternary]
        self.current_stage = 0

    def train(self, data_loader, num_epochs_per_stage=10):
        """多阶段训练"""
        for stage_idx, bits in enumerate(self.stages):
            self.current_stage = stage_idx

            # 设置当前量化精度
            self.set_quantization_bits(bits)

            # 训练当前阶段
            for epoch in range(num_epochs_per_stage):
                for batch in data_loader:
                    loss = self.training_step(batch)
                    self.optimizer.step()

            # 评估当前阶段
            acc = self.evaluate()
            print(f"Stage {stage_idx} ({bits}-bit): Accuracy = {acc:.4f}")

    def set_quantization_bits(self, bits):
        """设置量化位数"""
        if bits == 16:
            self.quantize = lambda x: x  # 无量化
        elif bits == 8:
            self.quantize = lambda x: self.round_to_int8(x)
        elif bits == 4:
            self.quantize = lambda x: self.round_to_int4(x)
        elif bits == 3:
            self.quantize = lambda x: self.ternarize(x)  # BWTA

    def ternarize(self, x):
        """三元化"""
        threshold = 0.5 * torch.std(x)
        return torch.where(x > threshold, 1.0,
               torch.where(x < -threshold, -1.0, 0.0))
```

### 性能

| 指标 | 数值 |
|------|------|
| Kernel 级别加速 | 16-24× |
| 端到端吞吐量 | 216-330 tokens/s |
| BERT GLUE 平均下降 | -3.5% |

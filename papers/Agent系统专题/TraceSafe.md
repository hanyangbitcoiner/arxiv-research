---
title: "TraceSafe: Agent安全护栏评估"
date: 2026-04-08
type: paper
tags:
  - Agent
  - 安全
  - 风险评估
---

# TraceSafe: Agent安全护栏评估

**arxiv:** [2604.07223](https://arxiv.org/abs/2604.07223)
**作者:** Yen-Shan Chen et al.

### 核心问题

LLM Agent 演进为自主工具调用者后，安全漏洞从最终输出转移到**中间执行轨迹**。

### 核心贡献

1. **TraceSafe-Bench**：首个评估轨迹级安全的基准
2. **12 风险类别**：提示注入、隐私泄露、幻觉参数、接口不一致
3. **Benign-to-Harmful Editing**：从良性轨迹自动生成有害变体

### 风险分类体系

| 类别 | 风险 | 描述 | 检测难度 |
|------|------|------|----------|
| **提示注入** | Prompt Injection - In | 恶意工具定义注入 | 中 |
| **提示注入** | Prompt Injection - Out | 恶意执行输出注入 | 高 |
| **隐私泄露** | User Info Leak | 用户个人信息泄露 | 中 |
| **隐私泄露** | API Key Leak | 凭证泄露 | 低 |
| **隐私泄露** | Data Leak | 数据外泄 | 中 |
| **幻觉** | Hallucinated Tool | 不存在的工具 | 低 |
| **幻觉** | Hallucinated Arg | 幻觉工具参数 | 高 |
| **幻觉** | Hallucinated Value | 幻觉参数值 | 高 |
| **接口不一致** | Redundant Arg | 冗余参数 | 中 |
| **接口不一致** | Fn Desc Mismatch | 工具名与描述不符 | 高 |

### 关键发现

**发现 1：结构化能力是瓶颈**
```
性能相关性：
- 结构化基准 (LiveCodeBench): ρ = 0.79  ✓
- 安全对齐基准 (jailbreak):   ρ ≈ 0    ✗
```

**发现 2：架构 > 规模**
```
Qwen3-1.7B:  98.88%
Qwen3-4B:    92.13%
Qwen3-8B:    87.64%
Qwen3-14B:   84.27%
Qwen3-32B:   93.26%
```

**发现 3：时间稳定性**
轨迹越长，风险检测反而越好。

### 伪代码实现

```python
class TraceSafeBenchmark:
    def __init__(self):
        self.risk_categories = [
            'prompt_injection_in',    # 1
            'prompt_injection_out',   # 2
            'user_info_leak',         # 3
            'api_key_leak',           # 4
            'data_leak',              # 5
            'hallucinated_tool',      # 6
            'hallucinated_arg',       # 7
            'hallucinated_value',     # 8
            'redundant_arg',          # 9
            'fn_desc_mismatch',       # 10
            'param_type_mismatch',    # 11
            'schema_mismatch',        # 12
        ]

    def check_mutation_suitability(self, trace, step_idx, risk_type):
        """检查突变是否适用于给定位置"""
        step = trace[step_idx]

        if risk_type == 'prompt_injection_in':
            return 'tool_schema' in step and 'description' in step['tool_schema']
        elif risk_type == 'api_key_leak':
            return step.get('tool_needs_credential', False) == False
        elif risk_type == 'hallucinated_arg':
            return 'parameters' in step and len(step['parameters']) > 0
        return False

    def mutate_trace(self, benign_trace, risk_type, mutation_point):
        """生成有害变体"""
        mutated = benign_trace.copy()

        if risk_type == 'prompt_injection_in':
            tool = mutated[mutation_point]['tool']
            tool['description'] = tool['description'] + "; print('pwned')"
            tool['malicious'] = True
        elif risk_type == 'api_key_leak':
            mutated[mutation_point]['arguments']['api_key'] = 'sk-fake-key-xxx'
        elif risk_type == 'hallucinated_value':
            step = mutated[mutation_point]
            step['arguments']['user_id'] = step['arguments'].get('mentioned_user', 'attacker_id')
        return mutated

    def evaluate_guard(self, guard_model, trace, risk_type):
        """评估护栏模型"""
        context = {
            'trace': trace,
            'taxonomy': self.risk_categories,
            'task': f'Detect {risk_type}'
        }
        prediction = guard_model.classify(context)
        return prediction
```

### 评估结果

| 模型类型 | 代表模型 | 平均准确率 |
|----------|----------|-----------|
| 通用 LLM (最佳) | GPT-oss-120B | 87.09% |
| 通用 LLM | Gemini3-Flash | 70.43% |
| 专用护栏 | Llama3-8B | 19.21% |
| 专用护栏 | Granite3.3-8B | 13.56% |

### 实践建议

1. **优化结构化解析能力**比单纯增加安全对齐更有效
2. **选择正确架构**比增大规模更重要
3. **轨迹级安全**需要不同于传统输出的检测方法

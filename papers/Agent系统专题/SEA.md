---
title: "SEA: 自学习诊断Agent"
date: 2026-04-08
type: paper
tags:
  - Agent
  - 记忆
  - 强化学习
---

# SEA: 自学习诊断Agent

**arxiv:** [2604.07269](https://arxiv.org/abs/2604.07269)
**作者:** Bingxuan Li, Simo Du, Yue Guo

### 核心问题

临床诊断专家依赖从经验中形成的**可复用诊断模式**。现有 Agent 存在：
- 每个案例独立处理，无法跨案例复用经验
- 无法从反馈中自动改进

### 核心贡献

1. **双记忆架构**：短程记忆（案例）+ 长程记忆（规则）
2. **联合强化学习框架**：同时优化诊断能力和记忆管理
3. **可插拔设计**：无需修改模型参数即可提升性能

### 双记忆架构

```
Agent Policy Model
输入: 患者病例 xt
输出: 诊断结果 ot + 记忆操作 ut
         │
┌────────┴────────┐
▼                   ▼
短程记忆 MS        长程记忆 ML
- 最近 K 个案例    - 抽象诊断规则
- append-only     - 从经验中提炼
```

**数学形式：**
```
MS^t = {c1, ..., c|MS^t|}, |MS^t| ≤ K
ML^t = {r1, ..., r|ML^t|}
```

### 奖励函数

```python
def compute_reward(action, diagnosis_correct, round_t, T):
    r_diag = 5 if diagnosis_correct else -5
    r_mem = -3 * len(self.MS) / self.K

    # 早期强调记忆形成，后期强调诊断
    lambda_diag = 1.0 * (round_t / T)
    lambda_mem = 1.0 * (1 - round_t / T)

    return lambda_diag * r_diag + lambda_mem * r_mem
```

### 伪代码实现

```python
class DualMemoryAgent:
    def __init__(self, policy_model, K=10):
        self.policy = policy_model
        self.K = K
        self.MS = []  # 短程案例记忆
        self.ML = []  # 长程规则记忆

    def select_action(self, case, round_t, total_rounds):
        context = self.build_context(case)
        actions = ['list', 'append', 'pop', 'consolidate']
        probs = self.policy.compute_action_probs(context, actions)

        # 早期更倾向记忆操作
        if round_t < total_rounds * 0.5:
            probs['list'] *= 1.5
            probs['append'] *= 1.2
        return self.sample_action(probs)

    def update_memory(self, action, case, outcome):
        if action == 'append':
            self.MS.append({
                'case': case,
                'outcome': outcome,
                'prediction': outcome['pred'],
                'feedback': outcome['feedback']
            })
            if len(self.MS) > self.K:
                self.consolidate_oldest()
        elif action == 'consolidate':
            oldest = self.MS.pop(0)
            rule = self.summarize_to_rule(oldest)
            self.ML.append(rule)

    def consolidate_oldest(self):
        if not self.MS:
            return
        oldest = self.MS[0]
        rule = self.summarize_to_rule(oldest)
        self.ML.append(rule)
        self.MS.pop(0)

    def summarize_to_rule(self, case_record):
        prompt = f"""
        从以下案例中提炼一条诊断规则：
        症状: {case_record['case']['symptoms']}
        诊断: {case_record['outcome']['diagnosis']}
        """
        return {'rule': self.policy.generate(prompt), 'source': case_record}
```

### 实验结果

| 设置 | 方法 | 准确率 | 提升 |
|------|------|--------|------|
| 标准评估 | SEA (Qwen-8B) | **92.46%** | **+19.6%** |
| 长期任务 | SEA (Qwen-8B) | **72.14%** | **+0.35** |

### 关键洞察

1. **联合优化优于单独优化**：仅优化诊断正确率收益有限
2. **记忆需要结构**：简单添加记忆反而降低性能（ReAct+记忆 42.51% < 零样本 72.10%）
3. **可插拔性**：双记忆模块可搭配任意基座模型

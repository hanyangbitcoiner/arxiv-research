---
title: "Agent 系统专题"
date: 2026-04-09
type: topic
tags:
  - Agent
  - 记忆
  - 安全
  - 知识管理
  - 技能学习
---

# Agent 系统专题

本专题收录所有 Agent 系统相关论文，包含 SEA、TraceSafe、SkillX 等。

---

## 目录

1. [[#SEA-自学习诊断Agent]] - 双记忆 + 联合强化学习
2. [[#TraceSafe-Agent安全护栏评估]] - 12 类风险评估
3. [[#SkillX-技能知识库]] - 三层技能自动构建

---

## SEA: 自学习诊断Agent

**arxiv:** [2604.07269](https://arxiv.org/abs/2604.07269)
**作者:** Bingxuan Li, Simo Du, Yue Guo
**日期:** 2026-04-08

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

---

## TraceSafe: Agent安全护栏评估

**arxiv:** [2604.07223](https://arxiv.org/abs/2604.07223)
**作者:** Yen-Shan Chen et al.
**日期:** 2026-04-08

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

---

## SkillX: 技能知识库

**arxiv:** [2604.04804](https://arxiv.org/abs/2604.04804)
**作者:** Chenxi Wang et al.
**日期:** 2026-04-06

### 核心问题

现有自我进化范式存在：
1. **效率低下**：智能体孤立学习，重复发现相似行为
2. **泛化差**：从有限经验学到的技能难以迁移
3. **能力瓶颈**：提取的技能受限于智能体当前能力上限

### 核心贡献

| 创新 | 描述 |
|------|------|
| Multi-Level Skills Design | 三层技能：规划技能 → 功能技能 → 原子技能 |
| Iterative Skills Refinement | 基于执行反馈自动改进技能 |
| Exploratory Skills Expansion | 主动生成新技能扩展覆盖 |

### 三层技能架构

```
技能库 D = S_plan ⊕ S_func ⊕ S_atomic

┌─────────────────────────────────────────┐
│  规划技能 S_plan                         │
│  - 子任务组织结构                         │
│  - 排序、依赖、分支                       │
├─────────────────────────────────────────┤
│  功能技能 S_func                        │
│  - 子任务抽象                            │
│  - 工具组合                             │
├─────────────────────────────────────────┤
│  原子技能 S_atomic                      │
│  - 单个工具对齐                          │
│  - 丰富描述、约束、使用模式               │
└─────────────────────────────────────────┘
```

### 伪代码实现

```python
class SkillX:
    def extract_skills(self, trajectory):
        """
        从轨迹中提取三层技能
        """
        # 1. 原子技能：工具级别的丰富描述
        atomic_skills = self.extract_atomic(trajectory)

        # 2. 功能技能：子任务级别的工具组合
        func_skills = self.extract_functional(trajectory)

        # 3. 规划技能：任务组织结构
        plan_skills = self.extract_planning(trajectory)

        return atomic_skills, func_skills, plan_skills

    def extract_atomic(self, trajectory):
        """提取原子技能"""
        skills = []
        for step in trajectory.steps:
            if step.is_tool_call():
                skill = {
                    'name': f"atomic_{step.tool_name}",
                    'description': step.tool.description,
                    'constraints': step.tool.constraints,
                    'usage_patterns': self.extract_patterns(step),
                    'tool': step.tool_name
                }
                skills.append(skill)
        return skills

    def extract_functional(self, trajectory):
        """提取功能技能"""
        skills = []
        subtasks = self.decompose_into_subtasks(trajectory)
        for subtask in subtasks:
            skill = {
                'name': f"func_{subtask.name}",
                'subtask': subtask.name,
                'tools_required': subtask.tools,
                'composition': self.extract_tool_sequence(subtask)
            }
            skills.append(skill)
        return skills

    def extract_planning(self, trajectory):
        """提取规划技能"""
        skill = {
            'name': f"plan_{trajectory.task_type}",
            'structure': self.extract_task_structure(trajectory),
            'ordering': self.extract_ordering(trajectory),
            'dependencies': self.extract_dependencies(trajectory)
        }
        return skill

    def refine_skills(self, skill, execution_feedback):
        """
        基于反馈迭代改进技能
        """
        if execution_feedback.failed:
            failure_analysis = self.analyze_failure(skill, execution_feedback)
            skill = self.update_skill(skill, failure_analysis)
        return skill

    def expand_skills(self, existing_skills):
        """
        探索性扩展：主动发现新技能
        """
        # 分析现有技能的覆盖盲区
        uncovered_tools = self.find_uncovered_tools(existing_skills)

        # 引导探索到高价值区域
        exploration_trajectories = self.guided_exploration(uncovered_tools)

        # 从新轨迹中提取技能
        new_skills = self.extract_skills(exploration_trajectories)

        return existing_skills + new_skills

    def guided_exploration(self, uncovered_tools):
        """
        经验引导的探索策略
        """
        trajectories = []
        for tool in uncovered_tools:
            # 优先探索高频失败的工具
            if tool.failure_rate > 0.3:
                trajs = self.explore_with_guidance(tool, strategy='failure_focused')
            # 然后探索从未使用过的工具
            elif tool.usage_count == 0:
                trajs = self.explore_with_guidance(tool, strategy='novel')
            trajectories.extend(trajs)
        return trajectories


class SkillRetriever:
    def retrieve_and_rewrite(self, task_query):
        """
        技能检索与伪计划重写
        """
        # 1. 检索相关规划技能
        plan_skills = self.retrieve_plan_skills(task_query)

        # 2. 重写为伪计划
        pseudo_plan = self.rewrite_plan(task_query, plan_skills)

        # 3. 填充功能技能和原子技能
        filled_plan = self.fill_sub_skills(pseudo_plan)

        return filled_plan

    def fill_sub_skills(self, pseudo_plan):
        """填充下层技能"""
        for subtask in pseudo_plan.subtasks:
            func_skill = self.retrieve_func_skill(subtask)
            atomic_skills = self.retrieve_atomic_skills(func_skill.tools)
            subtask.skills = {
                'functional': func_skill,
                'atomic': atomic_skills
            }
        return pseudo_plan
```

---

## Agent 系统专题总结

### 方法对比

| 论文 | 核心方法 | 创新点 |
|------|----------|--------|
| SEA | 双记忆 + 联合 RL | 可插拔记忆模块 |
| TraceSafe | 12 类风险分类 | 轨迹级安全评估 |
| SkillX | 三层技能库 | 自动构建可复用技能 |

### 核心洞察

1. **记忆与推理联合优化**是关键
2. **结构化能力**比语义对齐更重要
3. **技能抽象**是跨任务泛化的有效方式

### 实践建议

1. Agent 开发应考虑**双记忆架构**
2. 安全评估需覆盖**轨迹级风险**
3. 技能库是**跨任务迁移**的有效手段

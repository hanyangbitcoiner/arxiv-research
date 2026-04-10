---
title: "SkillX: 技能知识库"
date: 2026-04-06
type: paper
tags:
  - Agent
  - 知识管理
  - 技能学习
---

# SkillX: 技能知识库

**arxiv:** [2604.04804](https://arxiv.org/abs/2604.04804)
**作者:** Chenxi Wang et al.

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

---
title: "论文索引"
date: 2026-04-09
type: index
tags:
  - 索引
  - 论文
---

# 论文索引

本目录包含每篇论文的独立深度解读文件，点击标签可跳转相关论文。

## 按领域分类

### Agent 系统

| 论文 | 标签 | 关键方法 |
|------|------|----------|
| [[2604.07269-SEA-双记忆Agent\|SEA]] | `#双记忆` `#强化学习` | 双记忆 + 联合 RL |
| [[2604.04804-SkillX-技能知识库\|SkillX]] | `#技能知识库` `#经验学习` | 三层技能库自动构建 |
| [[2604.07223-TraceSafe-安全护栏\|TraceSafe]] | `#安全护栏` `#风险评估` | 12 类风险分类 |

### 模型量化

| 论文 | 标签 | 关键方法 |
|------|------|----------|
| [[2604.06798-MoBiE-MoE二值化\|MoBiE]] | `#MoE` `#二值化` | CEJD + GLAS + NGES |
| [[2604.03957-BWTA-二值化Transformer\|BWTA]] | `#Transformer` `#CUDA` | 三值激活 + CUDA Kernel |

### 推理优化

| 论文 | 标签 | 关键方法 |
|------|------|----------|
| [[2604.04987-Cactus-SpeculativeDecoding\|Cactus]] | `#SpeculativeDecoding` `#约束优化` | 受控偏离投机采样 |

### 训练数据

| 论文 | 标签 | 关键方法 |
|------|------|----------|
| [[2604.05114-pi2-长上下文推理\|π²]] | `#长上下文` `#表格` | 表格扩展 + 反向翻译 |

---

## 按标签浏览

### 技术标签

| 标签 | 论文数量 | 论文 |
|------|----------|------|
| [[Agent]] | 3 | SEA, SkillX, TraceSafe |
| [[二值化]] | 2 | MoBiE, BWTA |
| [[量化]] | 2 | MoBiE, BWTA |
| [[推理加速]] | 1 | Cactus |
| [[长上下文]] | 1 | π² |
| [[安全护栏]] | 1 | TraceSafe |

### 方法标签

| 标签 | 论文数量 | 论文 |
|------|----------|------|
| [[强化学习]] | 1 | SEA |
| [[双记忆]] | 1 | SEA |
| [[技能知识库]] | 1 | SkillX |
| [[SpeculativeDecoding]] | 1 | Cactus |
| [[约束优化]] | 1 | Cactus |
| [[CUDA]] | 1 | BWTA |

---

## Obsidian 使用提示

1. **点击链接跳转**：使用 `[[文件名]]` 双向链接
2. **按标签筛选**：在 Obsidian 中搜索 `tag:#标签名`
3. **Dataview 查询**：

```dataview
TABLE title, date, arxiv
FROM "papers/md"
WHERE type = "paper"
SORT date DESC
```

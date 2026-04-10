---
title: "π²: 长上下文推理数据构建"
date: 2026-04-06
type: paper
tags:
  - 长上下文
  - 推理
  - 训练数据
---

# π²: 长上下文推理数据构建

**arxiv:** [2604.05114](https://arxiv.org/abs/2604.05114)
**作者:** Quyet V. Do et al. (Virginia Tech)

### 核心问题

长上下文推理能力难以提升，缺乏高质量、结构化的推理训练数据。

**现有数据的问题：**
1. **缺乏真实多样性**：多是单跳问题，缺乏多跳推理
2. **上下文覆盖不足**：单文档单表格，难以测试跨文档理解
3. **推理轨迹缺失**：没有结构化的中间步骤

### 核心贡献

π² 流水线构建高质量推理数据：

```
┌─────────────────────────────────────────────────────────┐
│                    π² 数据构建流水线                      │
├─────────────────────────────────────────────────────────┤
│  Step 1: 表格收集与扩展                                  │
│  - 从维基百科提取表格                                    │
│  - 合成扩展新列（基于外部知识）                            │
├─────────────────────────────────────────────────────────┤
│  Step 2: QA 对生成                                      │
│  - 多跳分析推理问题                                      │
│  - SQL + Python 双重执行验证                             │
├─────────────────────────────────────────────────────────┤
│  Step 3: 推理轨迹生成                                   │
│  - 反向翻译生成结构化推理步骤                              │
└─────────────────────────────────────────────────────────┘
```

### 方法详解

#### Step 1: 表格收集与扩展

```python
class TableCollector:
    """
    从维基百科收集表格并扩展
    """

    def __init__(self, min_rows=5, max_rows=30, min_cols=2, max_cols=6):
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.max_cols = max_cols

    def extract_from_wikipedia(self, page_titles):
        """
        从维基百科页面提取表格
        """
        tables = []
        for title in page_titles:
            page_content = self.fetch_wikipedia_page(title)
            page_tables = self.parse_tables(page_content)

            # 过滤大小合适的表格
            for table in page_tables:
                if self.is_valid_size(table):
                    tables.append(table)

        return tables

    def is_valid_size(self, table):
        """表格大小过滤"""
        rows, cols = table.shape
        return (self.min_rows <= rows <= self.max_rows and
                self.min_cols <= cols <= self.max_cols)


class TableExpander:
    """
    表格扩展：增加新列超越原文内容
    """

    def __init__(self, max_expansion=3):
        self.max_expansion = max_expansion

    def expand_table(self, table, wikipedia_context):
        """
        合成表格扩展

        策略：
        1. 如果某列包含外部链接，提取链接页面的信息作为新列
        2. 使用 LLM 生成与现有列语义相关的新列
        """
        expanded = table.copy()
        original_columns = table.columns

        # 尝试为每个可扩展的列添加新列
        for col_idx, column in enumerate(original_columns):
            if self.can_expand(column):
                # 提取链接页面的信息
                if column.has_external_links():
                    new_column = self.extract_from_links(column)
                else:
                    # 使用 LLM 生成
                    new_column = self.generate_column(column, table)

                if self.verify_column(new_column, table):
                    expanded.add_column(new_column)
                    if len(expanded.columns) - len(original_columns) >= self.max_expansion:
                        break

        return expanded

    def extract_from_links(self, column):
        """从外部链接提取信息"""
        linked_pages = column.get_linked_pages()
        new_values = []

        for page in linked_pages:
            page_content = self.fetch_page_content(page)
            info = self.extract_relevant_info(page_content, column)
            new_values.append(info)

        return Column(name=f"{column.name}_expanded", values=new_values)

    def generate_column(self, column, table):
        """使用 LLM 生成新列"""
        prompt = f"""
        给定表格：
        {table.to_string()}

        列 "{column.name}" 包含: {column.values[:5]}

        生成一个与现有列语义相关的新列。
        新列应该包含真实的信息，不是简单的重复。

        格式：
        列名: [generated_column_name]
        值: [value1, value2, value3, ...]
        """

        response = self.llm.generate(prompt)
        return self.parse_column_response(response)
```

#### Step 2: QA 对生成

```python
class QAGenerator:
    """
    多跳推理问答对生成
    """

    def __init__(self):
        self.question_types = ['aggregation', 'comparison', 'entity']

    def generate_qa_pairs(self, table, num_questions=5):
        """
        为表格生成多样化的问答对
        """
        qa_pairs = []

        for _ in range(num_questions):
            # 选择问题类型
            q_type = random.choice(self.question_types)

            if q_type == 'aggregation':
                qa = self.generate_aggregation_question(table)
            elif q_type == 'comparison':
                qa = self.generate_comparison_question(table)
            else:  # entity
                qa = self.generate_entity_question(table)

            # 验证答案正确性
            if self.verify_answer(qa, table):
                qa_pairs.append(qa)

        return qa_pairs

    def generate_aggregation_question(self, table):
        """
        生成聚合类问题
        例如：某列的总和、平均值、计数等
        """
        prompt = f"""
        给定表格：
        {table.to_string()}

        生成一个需要计数或求和的聚合问题。

        格式：
        问题: [问题内容]
        答案: [数值]
        SQL: [用于验证的SQL查询]
        """

        response = self.llm.generate(prompt)
        return self.parse_qa_response(response)

    def generate_comparison_question(self, table):
        """
        生成比较类问题
        例如：谁最多、哪个最小等
        """
        prompt = f"""
        给定表格：
        {table.to_string()}

        生成一个需要比较的问题。

        格式：
        问题: [问题内容]
        答案: [实体名称]
        验证: [比较依据]
        """

        return self.parse_qa_response(self.llm.generate(prompt))

    def verify_answer(self, qa, table):
        """
        使用双重执行验证答案
        1. SQL 查询
        2. Python 计算
        """
        try:
            # 方法1: SQL 查询
            sql_result = self.execute_sql(qa['sql'], table)

            # 方法2: Python 计算
            python_result = self.execute_python(qa['code'], table)

            # 结果必须一致
            return self.results_match(sql_result, python_result)
        except:
            return False
```

#### Step 3: 推理轨迹生成

```python
class ReasoningTraceGenerator:
    """
    反向翻译生成结构化推理轨迹
    """

    def generate_trace(self, question, answer, context):
        """
        给定问答对和上下文，生成逐步推理轨迹

        关键洞察：
        - 使用反向翻译而非直接生成
        - 保证答案的正确性
        - 中间步骤可验证
        """
        prompt = f"""
        给定：
        - 问题: {question}
        - 答案: {answer}
        - 上下文: {context}

        生成逐步推理轨迹。

        要求：
        1. 每个步骤应该清晰、可验证
        2. 步骤之间有逻辑联系
        3. 最终得出正确答案

        格式：
        Step 1: [动作/观察]
        Step 2: [动作/观察]
        ...
        Step N: [最终推理，得出答案]
        """

        return self.llm.generate(prompt)

    def backtranslate_trace(self, question, answer, context, num_attempts=3):
        """
        反向翻译：从答案生成多个可能的推理轨迹

        选择最合理的一个
        """
        traces = []
        for _ in range(num_attempts):
            trace = self.generate_trace(question, answer, context)

            # 验证轨迹是否合理
            if self.verify_trace(trace, question, answer):
                traces.append(trace)

        # 选择最详细的轨迹
        return max(traces, key=lambda t: len(t.split('Step')))

    def verify_trace(self, trace, question, answer):
        """验证推理轨迹的正确性和完整性"""
        # 1. 包含足够多的步骤
        if trace.count('Step') < 2:
            return False

        # 2. 最终步骤包含答案相关词汇
        final_step = trace.split('Step')[-1]
        if not any(word in final_step.lower() for word in answer.lower().split()):
            return False

        # 3. 可重复性：重新执行能得到相同答案
        return True
```

### QA 类型示例

| 类型 | 问题示例 | 涉及操作 |
|------|----------|----------|
| Aggregation | "Debbie Reynolds 在 NBC 播出的节目中扮演了多少个不同角色？" | COUNT, WHERE |
| Comparison | "2023 年板球世界杯中，谁的击球手得分最多？" | ORDER BY, LIMIT |
| Entity | "哪个国家的钴储量与年产量比最高？" | 计算比率 |

### 实验结果

| 模型 | 基准 | 提升 |
|------|------|------|
| GPT-OSS-20B | 4 个长上下文基准 | +4.3% |
| QWEN3-4B-INSTRUCT | 4 个长上下文基准 | +2.7% |
| 自蒸馏 | GPT-OSS-20B | +4.4% |

### 核心洞察

1. **表格是好的数据源**：结构化、真实、适合多跳推理
2. **扩展超越原文**：避免模型记忆，直接测试推理能力
3. **反向翻译保证正确性**：从答案反推步骤

---
name: arxiv-research
description: 自动获取 arxiv LLM 论文并生成中文研究报告
argument-hint: <研究主题，如 "LLM 推理优化" 或 "LLM Agent 研究">
level: 3
---

# Arxiv 论文研究技能

自动从 arxiv 获取指定主题的最新论文，解析摘要，分析趋势，并生成中文研究报告。

## 功能

1. 通过 arxiv API 获取指定主题的论文
2. 解析论文标题、作者、摘要、日期
3. 按研究方向分类整理
4. 生成 Obsidian 友好的中文报告（带 front-matter 和双向链接）

## 使用方式

```
/arxiv-research LLM 推理优化最新进展
/arxiv-research RAG 和 Agent 系统
/arxiv-researchtransformer 架构改进
```

## 报告格式

生成的报告包含：
- **Front-matter**：Obsidian 兼容的 tags、日期、类型
- **执行摘要**：核心发现概览
- **分类论文表**：按研究方向整理的论文列表
- **关键发现**：每篇重要论文的详细分析
- **跨领域主题**：跨研究方向的主题分析
- **实践建议**：基于论文的可行建议

## 研究方向

当指定 "LLM" 泛指时，默认覆盖四个方向：
1. **评估基准** (benchmarks)：benchmark、evaluation、performance
2. **推理优化** (optimization)：quantization、distillation、inference
3. **Agent/RAG** (agents)：agent、RAG、retrieval、tool、memory
4. **架构改进** (architecture)：architecture、attention、transformer、context

## 输出路径

报告保存至：`{project}/arxiv-llm-research-{YYYY-MM-DD}.md`
原始数据保存至：`{project}/.omc/research/arxiv-{date}/`

## 脚本位置

配套脚本：`scripts/arxiv-llm-weekly-report.py`

```bash
# 命令行用法
python scripts/arxiv-llm-weekly-report.py                    # 生成周报
python scripts/arxiv-llm-weekly-report.py --days 30          # 月报
python scripts/arxiv-llm-weekly-report.py --output custom.md # 自定义输出
```

## 执行流程

1. 解析用户输入的研究主题
2. 构建 arxiv 查询语句
3. 调用 arxiv API 获取论文列表
4. 解析 XML 响应提取论文元数据
5. 按研究方向分类
6. 生成中文报告
7. 保存到项目目录

## 注意事项

- API 请求有频率限制，每次查询间隔建议 3 秒
- 报告自动去重，同一篇论文不会重复出现
- 使用中文撰写所有分析内容

# Arxiv Research

自动获取 arxiv LLM 论文并生成中文研究报告的 Claude Code Skill。

## 功能

- 通过 arxiv API 获取指定主题的最新论文
- 解析论文标题、作者、摘要、日期
- 按研究方向分类整理（评估基准、推理优化、Agent/RAG、架构改进）
- 生成 Obsidian 友好的中文报告（带 front-matter 和双向链接）

## 使用方式

### Claude Code 中使用

```
/arxiv-research LLM 推理优化最新进展
/arxiv-research RAG 和 Agent 系统
```

### 命令行使用

```bash
# 安装依赖
pip install urllib3

# 生成周报（默认最近7天）
python scripts/arxiv-llm-weekly-report.py

# 生成月报（最近30天）
python scripts/arxiv-llm-weekly-report.py --days 30

# 指定输出文件名
python scripts/arxiv-llm-weekly-report.py --output my-report.md
```

## 目录结构

```
.
├── scripts/
│   └── arxiv-llm-weekly-report.py    # 周报生成脚本
├── skills/
│   └── arxiv-research/
│       └── SKILL.md                  # Claude Code Skill 定义
├── CLAUDE.md                         # Claude Code 项目配置
└── README.md
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

默认覆盖四个方向：
1. **评估基准** (benchmarks)：benchmark、evaluation、performance
2. **推理优化** (optimization)：quantization、distillation、inference
3. **Agent/RAG** (agents)：agent、RAG、retrieval、tool、memory
4. **架构改进** (architecture)：architecture、attention、transformer、context

## 数据来源

- **API**: export.arxiv.org
- **类别**: cs.CL, cs.LG, cs.AI, cs.CR

## License

MIT

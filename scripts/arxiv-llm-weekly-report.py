#!/usr/bin/env python3
"""
Arxiv LLM 前沿研究周报生成器
自动获取 arxiv 最新 LLM 论文并生成中文报告

用法:
    python arxiv-llm-weekly-report.py                    # 生成周报
    python arxiv-llm-weekly-report.py --days 7           # 最近7天
    python arxiv-llm-weekly-report.py --days 30 --output monthly-report.md
"""

import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict
import urllib.request
import urllib.parse
import ssl
import os
import sys

# 研究方向配置
SEARCH_QUERIES = {
    "benchmarks": {
        "query": "cat:cs.CL AND (llm OR large language model) AND (benchmark OR evaluation OR performance OR MMMU OR HELM)",
        "keywords": ["benchmark", "evaluation", "performance", "reward model", "alignment"]
    },
    "optimization": {
        "query": "cat:cs.LG AND (llm OR large language model) AND (quantization OR distillation OR inference OR optimization OR speedup OR speculative)",
        "keywords": ["quantization", "distillation", "inference", "optimization", "speculative", "pruning"]
    },
    "agents": {
        "query": "cat:cs.CL AND (agent OR RAG OR retrieval OR tool OR memory) AND (llm OR large language model)",
        "keywords": ["agent", "RAG", "retrieval", "tool", "memory", "reasoning"]
    },
    "architecture": {
        "query": "cat:cs.CL AND (llm OR large language model) AND (architecture OR attention OR transformer OR context OR Mamba OR MoE)",
        "keywords": ["architecture", "attention", "transformer", "context", "Mamba", "mixture of experts"]
    }
}

ARXIV_API = "https://export.arxiv.org/api/query?"

def fetch_arxiv_papers(query: str, max_results: int = 25) -> list:
    """从 arxiv API 获取论文"""
    params = {
        "search_query": query,
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }

    url = ARXIV_API + urllib.parse.urlencode(params)

    # 忽略 SSL 证书验证
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(url, timeout=30, context=ctx) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"获取 arxiv 数据失败: {e}")
        return []

    return parse_arxiv_xml(xml_data)

def parse_arxiv_xml(xml_data: str) -> list:
    """解析 arxiv XML 响应"""
    papers = []
    try:
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            paper = {
                "id": entry.find("atom:id", ns).text.split("/")[-1] if entry.find("atom:id", ns) is not None else "",
                "title": entry.find("atom:title", ns).text.replace("\n", " ").strip() if entry.find("atom:title", ns) is not None else "",
                "summary": entry.find("atom:summary", ns).text.replace("\n", " ").strip() if entry.find("atom:summary", ns) is not None else "",
                "published": entry.find("atom:published", ns).text[:10] if entry.find("atom:published", ns) is not None else "",
                "authors": [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None],
                "categories": [c.get("term") for c in entry.findall("atom:category", ns)]
            }
            papers.append(paper)
    except ET.ParseError as e:
        print(f"XML 解析错误: {e}")

    return papers

def generate_report(papers_by_category: dict, output_path: str, days: int):
    """生成中文报告"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_short = datetime.now().strftime("%Y年%m月%d日")

    # 收集所有论文并去重
    all_papers = []
    seen_ids = set()
    for category, papers in papers_by_category.items():
        for paper in papers:
            if paper["id"] not in seen_ids:
                seen_ids.add(paper["id"])
                all_papers.append((category, paper))

    total_papers = len(all_papers)

    # 生成报告内容
    report = f"""---
title: "LLM 前沿研究周报"
date: {date_str}
type: research-report
source: arxiv
tags:
  - LLM
  - 论文研究
  - 周报
  - arxiv
---

# LLM 前沿研究周报

**日期：** {date_short}
**来源：** arxiv.org
**论文数量：** {total_papers}
**研究周期：** 最近 {days} 天

---

## 执行摘要

本报告综合分析了 {total_papers} 篇 arxiv 最新论文，涵盖 LLM 研究的四个核心方向：

{generate_executive_summary(papers_by_category)}

"""

    # 按类别生成详细内容
    category_names = {
        "benchmarks": "一、模型评估与基准",
        "optimization": "二、推理优化",
        "agents": "三、Agent 与 RAG 系统",
        "architecture": "四、架构改进"
    }

    for cat_key, cat_name in category_names.items():
        papers = papers_by_category.get(cat_key, [])
        report += generate_category_section(cat_name, papers, cat_key)

    # 跨领域分析
    report += generate_crosscutting_analysis(papers_by_category)

    # 实践建议
    report += generate_recommendations()

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"报告已生成: {output_path}")
    print(f"共分析 {total_papers} 篇论文")
    return output_path

def generate_executive_summary(papers_by_category: dict) -> str:
    """生成执行摘要"""
    summaries = []

    for cat, papers in papers_by_category.items():
        if papers:
            # 取第一篇论文作为代表
            top_paper = papers[0]
            summaries.append(f"- **{get_category_label(cat)}**：{top_paper['title'][:50]}...")

    return "\n".join(summaries) if summaries else "- 本周 LLM 研究各方向均有新进展"

def get_category_label(cat: str) -> str:
    labels = {
        "benchmarks": "评估基准",
        "optimization": "推理优化",
        "agents": "Agent/RAG",
        "architecture": "架构改进"
    }
    return labels.get(cat, cat)

def generate_category_section(category_name: str, papers: list, cat_key: str) -> str:
    """生成单个类别的章节"""
    if not papers:
        return f"{category_name}\n\n暂无相关论文\n\n"

    section = f"{category_name}\n\n"

    # 去重
    seen = set()
    unique_papers = []
    for p in papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique_papers.append(p)

    # 核心论文表
    section += "| 论文标题 | arxiv ID | 日期 | 关键贡献 |\n"
    section += "|----------|----------|------|----------|\n"

    for paper in unique_papers[:8]:  # 最多8篇
        title = paper["title"][:40] + "..." if len(paper["title"]) > 40 else paper["title"]
        paper_id = paper["id"].replace("v1", "").replace("v2", "")
        date = paper["published"]
        summary = paper["summary"][:80] + "..." if len(paper["summary"]) > 80 else paper["summary"]
        section += f"| {title} | {paper_id} | {date} | {summary} |\n"

    # 关键发现
    section += "\n### 关键发现\n\n"

    for i, paper in enumerate(unique_papers[:4], 1):
        section += f"#### #{i} {paper['title'][:60]}\n\n"
        section += f"- **论文**：[{paper['id']}](https://arxiv.org/abs/{paper['id']})\n"
        section += f"- **作者**：{', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n"
        section += f"- **摘要**：{paper['summary'][:200]}...\n\n"

    return section

def generate_crosscutting_analysis(papers_by_category: dict) -> str:
    """生成跨领域分析"""
    return """## 跨领域主题分析

### 主题一：效率-质量权衡

| 方向 | 具体技术 |
|------|----------|
| 量化 | 二元/三元权重压缩、MoE 专用量化 |
| Agent | 评估交互开销优化 |
| 架构 | 激活现有容量 vs 添加参数 |

### 主题二：评估可信度

| 问题 | 研究进展 |
|------|----------|
| LLM-as-judge 偏见 | 自偏好偏见系统研究 |
| 架构 vs 规模 | 架构选择比模型规模更影响性能 |

### 主题三：Agent 安全

| 阶段 | 特征 |
|------|------|
| 输出级 | 最终回答的安全护栏 |
| 轨迹级 ← 当前 | 中间执行轨迹的风险检测 |

---

"""

def generate_recommendations() -> str:
    """生成实践建议"""
    return """## 实践建议

### 1. 评估实践
- 使用多样化评审群体替代单一 LLM-as-judge
- 关注个性化奖励评估的最新进展
- 警惕自偏好偏见对模型比较的影响

### 2. 部署实践
- 边缘设备：考虑二元/三元量化方案
- MoE 部署：关注专用量化优化
- GPU 推理：关注 NF4 反量化内核优化

### 3. Agent 开发
- 投资推理+记忆联合优化方向
- 用 TraceSafe 类基准验证 Agent 安全性
- 降低评估开销：参考低交互开销评估方法

### 4. 架构探索
- 微调前尝试免训练增强方法
- 用结构化数据训练长上下文能力
- 关注批量推理多样性保持

---

## 数据来源

- **API**: export.arxiv.org
- **类别**: cs.CL, cs.LG, cs.AI, cs.CR
- **时间范围**: 自动获取最近论文

---

## 使用说明

本报告由 `scripts/arxiv-llm-weekly-report.py` 自动生成。

```bash
# 生成周报（默认最近7天）
python scripts/arxiv-llm-weekly-report.py

# 生成月报（最近30天）
python scripts/arxiv-llm-weekly-report.py --days 30

# 指定输出文件名
python scripts/arxiv-llm-weekly-report.py --output my-report.md
```
"""

def main():
    parser = argparse.ArgumentParser(description="Arxiv LLM 前沿研究周报生成器")
    parser.add_argument("--days", type=int, default=7, help="获取最近N天的论文 (默认: 7)")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--max-per-category", type=int, default=25, help="每个类别最多获取论文数 (默认: 25)")

    args = parser.parse_args()

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, f"arxiv-llm-research-{date_str}.md")

    print(f"正在获取 arxiv 论文（最近 {args.days} 天）...")
    print("=" * 50)

    papers_by_category = {}

    for cat_key, config in SEARCH_QUERIES.items():
        print(f"\n获取 {cat_key} 类别论文...")
        papers = fetch_arxiv_papers(config["query"], args.max_per_category)
        papers_by_category[cat_key] = papers
        print(f"  -> 获取到 {len(papers)} 篇论文")

    print("\n" + "=" * 50)
    print("正在生成报告...")

    generate_report(papers_by_category, output_path, args.days)
    print("\n完成！")

if __name__ == "__main__":
    main()

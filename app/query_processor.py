"""
查询处理器 —— LLM 改写 + HyDE + Multi-Query + 查询分解 + 意图分类
同时保留硬编码同义词词典作为快速回退。
"""
import re
import logging
from typing import List, Dict, Set, Optional, Tuple

import jieba

logger = logging.getLogger(__name__)

# ---- 制造业同义词词典（快速回退） ----
SYNONYM_DICT: Dict[str, List[str]] = {
    "工资": ["薪资", "薪酬", "待遇", "收入", "薪水"],
    "请假": ["休假", "调休", "事假", "病假", "年假"],
    "SOP": ["标准操作", "操作规程", "作业指导", "操作手册"],
    "设备": ["机器", "装置", "仪器", "机台"],
    "故障": ["报错", "异常", "报警", "停机", "出错"],
    "维修": ["检修", "保养", "维护", "修理"],
    "质检": ["品控", "QC", "检验", "检测"],
    "生产": ["制造", "加工", "组装"],
    "安全": ["安规", "防护", "危险"],
    "培训": ["入职", "上岗", "学习"],
    "绩效": ["考核", "KPI", "评估"],
    "日报": ["报表", "报告", "日志"],
    "客户": ["甲方", "顾客", "订单"],
}

AGGREGATE_KEYWORDS: Set[str] = {
    "各", "所有", "全部", "最高", "最低", "平均",
    "合计", "总共", "分别", "排名", "排序", "汇总", "统计", "总计",
    "哪些", "哪个", "多少", "几个",
    "今天", "今日", "昨天", "昨日", "本周", "本月", "日报", "汇总",
}


def expand_query(query: str) -> List[str]:
    """硬编码同义词快速扩展。"""
    queries = [query]
    for keyword, synonyms in SYNONYM_DICT.items():
        if keyword in query:
            for syn in synonyms[:2]:
                expanded = query.replace(keyword, syn)
                if expanded != query:
                    queries.append(expanded)
    return queries


def extract_keywords(query: str, top_n: int = 8) -> List[str]:
    """jieba 关键词提取。"""
    tokens = list(set(jieba.lcut(query)))
    tokens.sort(key=lambda t: len(t), reverse=True)
    return [t for t in tokens if len(t) >= 1][:top_n]


def detect_intent(query: str) -> Dict[str, bool]:
    return {
        "is_aggregate": any(kw in query for kw in AGGREGATE_KEYWORDS),
        "is_factual": bool(re.search(r'[\d]+', query)),
        "is_table": any(kw in query for kw in ["表", "工资", "绩效", "考核", "日报", "进度"]),
        "is_conceptual": any(kw in query for kw in ["什么是", "如何", "怎么", "为什么", "原理"]),
    }


def is_aggregate_query(query: str) -> bool:
    return any(kw in query for kw in AGGREGATE_KEYWORDS)


# ======== LLM-based Query Processing ========

_HYDE_PROMPT = """你是一位企业知识库助手。请根据以下用户问题，写一段假设性的简短回答（3-5句话），
这段回答将用于在向量知识库中进行语义搜索。只输出假设回答，不要任何前缀或解释。

问题：{query}

假设回答："""


_MULTI_QUERY_PROMPT = """你是一个查询优化专家。请将以下用户问题改写为 {n} 个不同角度的查询，
以覆盖不同的表达方式和检索方向，提高召回率。

原始问题：{query}

要求：
1. 每个查询保持独立完整，可以单独用来搜索
2. 覆盖不同的关键词组合和表达方式
3. 如果可能，一个侧重精确匹配，一个侧重语义相近，一个侧重概括性提问

请直接输出 {n} 个查询，每行一个，不要编号："""


_QUERY_DECOMPOSE_PROMPT = """你是一个查询分析专家。判断以下问题是否需要分解为多个子问题来回答。
如果问题简单，输出 "SIMPLE"。如果需要分解，将问题拆解为 2-4 个子问题，每个子问题一行。

问题：{query}

输出（SIMPLE 或每行一个子问题）："""


_INTENT_CLASSIFY_PROMPT = """请分析以下用户问题的查询意图，输出 JSON 格式。

问题：{query}

输出 JSON（只输出 JSON，不要其他内容）：
{{"type": "factual|conceptual|aggregate|comparison|procedure", "complexity": "simple|medium|complex", "needs_decomposition": true|false, "keywords": ["kw1", "kw2"]}}"""


async def llm_query_rewrite(query: str, llm) -> str:
    """用 LLM 改写查询，扩展语义覆盖。"""
    try:
        prompt = f"请将以下用户问题改写为一个更完整、更具体的检索查询（保留原意，增加相关术语和同义词，25字以内）：\n\n{query}\n\n改写后的查询："
        resp = await llm.acomplete(prompt)
        rewritten = str(resp).strip()
        if len(rewritten) >= 2:
            logger.info("LLM 查询改写: %s → %s", query, rewritten)
            return rewritten
    except Exception as e:
        logger.warning("LLM 查询改写失败: %s", e)
    return query


async def generate_hyde(query: str, llm) -> Optional[str]:
    """生成 HyDE（假设文档嵌入）。"""
    try:
        prompt = _HYDE_PROMPT.format(query=query)
        resp = await llm.acomplete(prompt)
        hyde = str(resp).strip()
        if len(hyde) >= 10:
            logger.info("HyDE 生成: %d 字", len(hyde))
            return hyde
    except Exception as e:
        logger.warning("HyDE 生成失败: %s", e)
    return None


async def generate_multi_queries(query: str, llm, n: int = 3) -> List[str]:
    """生成多个查询变体。"""
    try:
        prompt = _MULTI_QUERY_PROMPT.format(query=query, n=n)
        resp = await llm.acomplete(prompt)
        queries = [line.strip() for line in str(resp).strip().split("\n")
                    if line.strip() and len(line.strip()) >= 3]
        valid = [q for q in queries if q != query][:n]
        if valid:
            logger.info("Multi-Query: 生成 %d 个变体", len(valid))
            return valid
    except Exception as e:
        logger.warning("Multi-Query 生成失败: %s", e)
    return []


async def decompose_query(query: str, llm) -> List[str]:
    """复杂查询分解为子问题。"""
    try:
        prompt = _QUERY_DECOMPOSE_PROMPT.format(query=query)
        resp = await llm.acomplete(prompt)
        text = str(resp).strip()
        if "SIMPLE" in text.upper():
            return [query]
        sub_queries = [line.strip() for line in text.split("\n")
                       if line.strip() and len(line.strip()) >= 3]
        if len(sub_queries) >= 2:
            logger.info("查询分解: %d 个子问题", len(sub_queries))
            return sub_queries
    except Exception as e:
        logger.warning("查询分解失败: %s", e)
    return [query]


async def classify_intent_llm(query: str, llm) -> dict:
    """LLM 意图分类。"""
    try:
        prompt = _INTENT_CLASSIFY_PROMPT.format(query=query)
        resp = await llm.acomplete(prompt)
        text = str(resp).strip()
        # 尝试提取 JSON
        import json
        m = re.search(r'\{[^}]+\}', text)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    # 回退到规则分类
    base = detect_intent(query)
    return {
        "type": "aggregate" if base["is_aggregate"] else ("conceptual" if base["is_conceptual"] else "factual"),
        "complexity": "complex" if base["is_aggregate"] else "medium",
        "needs_decomposition": base["is_aggregate"],
        "keywords": extract_keywords(query),
    }

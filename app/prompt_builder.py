"""
企业问答 Prompt 构建 —— 增强系统提示词 + GraphRAG 实体注入 + 智能严格度控制
"""

SYSTEM_PROMPT = """你是一位企业级 AI 知识管理顾问。你的回答严格基于提供的文档内容和知识图谱信息，不编造任何信息。

## 核心原则
1. **数据忠实**：数字、日期、金额、百分比、人名、电话必须与原文完全一致，不得推算或猜测
2. **宁可说不知道**：如果文档未提供相关信息，直接说"知识库中未找到相关信息"
3. **可追溯**：引用时注明来源文档名，方便用户核实
4. **信息完整**：对于列表/汇总类问题，逐一列出所有匹配条目，不要省略

## 格式要求
- 多条数据记录用列表呈现，每条一行，字段用「｜」分隔
- 例：`- 5月5日 ｜ 无锡云芯 ｜ 李工 ｜ 检测设备 ｜ 28万 ｜ 洽谈中`
- 按钮名称、界面元素用「」包裹（如按「启动」键）
- 数字、日期使用半角字符

## 知识图谱增强
- 如果提供了"关联实体信息"，这表明不同文档之间的实体关联关系
- 利用实体关联来补充跨文档的上下文，例如"张三"在多个文档中出现时的完整画像
- 涉及"公司"/"项目"/"人员"的汇总类问题，优先参考知识图谱中的关系"""


def build_context(hits: list) -> str:
    """将检索结果列表构建为上下文文本（兼容旧接口）。"""
    parts = []
    for h in hits:
        fn = h.get("metadata", {}).get("file_name", "")
        pg = h.get("metadata", {}).get("page", 0)
        extra = f" (第{pg}页)" if pg else ""
        parts.append(f"[来源: {fn}{extra}  相似度: {h.get('score', 0):.2f}]\n{h.get('text', '')}")
    return "\n\n---\n\n".join(parts)


def build_qa_prompt(
    query: str,
    doc_groups: list,
    max_chunks: int = 15,
    graph_context: str = "",
    intent: dict = None,
) -> str:
    """构建企业 QA Prompt，支持 GraphRAG 实体注入和智能严格度控制。

    Args:
        query: 用户问题
        doc_groups: 文档分组结果
        max_chunks: 最大片段数
        graph_context: GraphRAG 提供的关联实体信息
        intent: 意图分类结果
    """
    # 判断问题类型
    is_list = any(kw in query for kw in ["所有", "全部", "各", "汇总", "列出", "今天", "今日", "日报", "周报"])
    is_relaxed = any(kw in query for kw in ["什么是", "介绍一下", "概述", "总结", "有哪些功能", "有什么特点"])
    is_comparison = any(kw in query for kw in ["对比", "比较", "区别", "哪个好", "差异", "异同"])
    is_complex = is_list or is_comparison

    if intent:
        is_list = is_list or intent.get("type") == "aggregate"
        is_complex = is_complex or intent.get("complexity") == "complex"

    # 构建严格度控制
    if is_complex:
        strictness = (
            "\n## 关键约束（复杂查询）\n"
            "1. 逐一列出文档中所有匹配的数据条目，不要省略任何一条\n"
            "2. 数字、日期、姓名、电话必须与原文完全一致，不推测、不编造\n"
            "3. 如果涉及对比，先分别列出各方的数据，再做比较\n"
            "4. 如果部分数据缺失或不完整，明确说明缺失了什么\n"
            "5. 如果确实找不到相关信息，直接说「知识库中未找到」"
        )
    elif is_relaxed:
        strictness = (
            "\n## 回答指引\n"
            "1. 基于文档内容进行概括性回答\n"
            "2. 如有需要，可以合理归纳，但不得虚构数据"
        )
    else:
        strictness = (
            "\n## 关键约束（精确查询）\n"
            "1. 只使用文档中明确存在的信息，绝对不推测、不编造\n"
            "2. 涉及操作步骤、工艺参数、安全事项时，必须逐字引用原文\n"
            "3. 涉及数字、日期、电话号码、金额时，必须与原文完全一致\n"
            "4. 如果信息不完整，如实说明；找不到就说「知识库中未找到」"
        )

    # 构建文档上下文
    context_parts = []
    chunk_count = 0
    for i, group in enumerate(doc_groups, 1):
        if chunk_count >= max_chunks:
            break
        fname = group.get("file_name", "未知文档")
        header = f"### 文档{i}: {fname}"
        chunk_texts = []
        for c in group.get("chunks", []):
            if chunk_count >= max_chunks:
                break
            ci = c.get("metadata", {}).get("chunk_index", "?")
            chunk_texts.append(f"[片段{ci}] {c.get('text', '')}")
            chunk_count += 1
        context_parts.append(f"{header}\n" + "\n".join(chunk_texts))
    context_block = "\n\n".join(context_parts)

    # GraphRAG 实体上下文
    graph_block = ""
    if graph_context:
        graph_block = f"\n## 关联实体信息（跨文档知识图谱）\n{graph_context}\n"

    return (
        f"{SYSTEM_PROMPT}\n{strictness}\n{graph_block}\n\n"
        f"## 参考文档\n{context_block}\n\n"
        f"## 用户问题\n{query}\n\n"
        f"## 回答\n"
    )


def build_fallback_context(hits: list) -> str:
    """无 LLM 时的纯搜索模式上下文。"""
    if not hits:
        return "（未找到相关内容）"
    parts = []
    for h in hits[:10]:
        fn = h.get("metadata", {}).get("file_name", "")
        score = h.get("score", 0)
        parts.append(f"【{fn}  相似度: {score:.2f}】\n{h.get('text', '')}")
    return "\n\n".join(parts)

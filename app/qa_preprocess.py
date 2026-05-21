"""
QA 预处理 —— 借鉴 FastGPT：上传时为每个片段预生成问答对，提升搜索命中率。
原理：用户问"联系谁"可能不匹配原文"联系我们"，但会匹配预生成问题"怎么联系负责人"。
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


async def generate_qa_pairs(chunks: List[str], file_name: str) -> List[str]:
    """为每个片段预生成 2-3 个可能被问到的问题，加权重更高。
    返回扩充后的片段列表：原文 + 预生成问答。"""
    from app.rag_engine import get_llm
    llm = get_llm()
    if llm is None:
        return chunks  # 无 LLM 时跳过

    enriched = list(chunks)
    # 每 5 个片段批量生成一次，节省 API 调用
    batch_size = 5
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        texts = "\n\n---\n\n".join(
            f"[片段{i}] {c[:300]}" for i, c in enumerate(batch)
        )
        prompt = (
            f"以下是文档「{file_name}」的几个片段。"
            f"为每个片段生成 2 个用户可能用到的提问（简短口语化），"
            f"格式：Q1: ... | Q2: ... （每个片段一行，用换行分隔）\n\n{texts}"
        )
        try:
            resp = await llm.acomplete(prompt)
            for line in str(resp).strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                # 提取问题部分
                questions = []
                for part in line.split("|"):
                    part = part.strip()
                    if part.startswith("Q") and ":" in part:
                        q = part.split(":", 1)[1].strip()
                        if q and len(q) >= 4:
                            questions.append(q)
                for q in questions[:2]:
                    enriched.append(f"[自动问答] 问: {q}")
        except Exception as e:
            logger.warning("QA 预处理跳过 (batch %d): %s", batch_start, e)

    if len(enriched) > len(chunks):
        logger.info("QA 预处理: %d → %d 条 (文档: %s)",
                     len(chunks), len(enriched), file_name)
    return enriched

"""
高级分块策略 —— 语义分块 + Small-to-Big + 上下文富化 + 表格结构保留
"""
import re
import logging
import numpy as np
from typing import List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

_FILE_TYPE_LABELS = {
    ".pdf": "PDF文档", ".docx": "Word文档", ".doc": "Word文档",
    ".xlsx": "Excel表格", ".xls": "Excel表格",
    ".pptx": "PPT演示", ".ppt": "PPT演示",
    ".txt": "文本文件", ".md": "Markdown", ".csv": "CSV表格",
    ".png": "图片", ".jpg": "图片", ".jpeg": "图片",
}


def file_type_label(ext: str) -> str:
    return _FILE_TYPE_LABELS.get(ext.lower(), "文档")


def build_context_prefix(
    file_name: str,
    file_ext: str,
    sheet_name: str = "",
    table_header: str = "",
    page: int = 0,
    section: str = "",
) -> str:
    """构建片段上下文前缀，参与嵌入。"""
    parts = [
        f"[文档: {file_name}]",
        f"[类型: {file_type_label(file_ext)}]",
    ]
    if sheet_name:
        parts.append(f"[工作表: {sheet_name}]")
    if table_header:
        parts.append(f"[表头: {table_header}]")
    if page:
        parts.append(f"[页码: 第{page}页]")
    if section:
        parts.append(f"[章节: {section}]")
    return " ".join(parts) + "\n"


def enrich_chunks(
    chunks: List[str],
    file_name: str,
    file_ext: str,
    sheet_name: str = "",
    table_header: str = "",
    per_chunk_meta: Optional[List[dict]] = None,
) -> List[str]:
    """批量为片段添加上下文前缀。"""
    prefix = build_context_prefix(
        file_name, file_ext,
        sheet_name=sheet_name,
        table_header=table_header,
    )
    enriched = []
    for i, chunk in enumerate(chunks):
        extra = ""
        if per_chunk_meta and i < len(per_chunk_meta) and per_chunk_meta[i]:
            pg = per_chunk_meta[i].get("page", 0)
            if pg:
                pfx = build_context_prefix(
                    file_name, file_ext,
                    sheet_name=sheet_name,
                    table_header=table_header,
                    page=pg,
                )
                enriched.append(pfx + chunk)
                continue
        enriched.append(prefix + chunk)
    return enriched


def detect_sections(text: str) -> list:
    """检测中文文档的章节标题。"""
    patterns = [
        r'第[一二三四五六七八九十百千\d]+章\s*[^\n]*',
        r'第[一二三四五六七八九十百千\d]+节\s*[^\n]*',
        r'\d+[\.\、]\s*[^\n]{2,30}',
    ]
    sections = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            title = m.group().strip()
            if 2 <= len(title) <= 50:
                sections.append({"title": title, "pos": m.start()})
    sections.sort(key=lambda s: s["pos"])
    return sections


# ======== 语义分块 ========

def semantic_split(
    text: str,
    embed_model=None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    similarity_threshold: float = 0.6,
) -> List[str]:
    """基于语义相似度的自适应分块。

    原理：先按句子/段落分割，计算相邻文本块的语义相似度，
    在相似度突变处切分，保证每个块内部主题一致。

    如果 embed_model 为 None，退化为段落感知滑动窗口。
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    if not text or not text.strip():
        return []

    # 按句子分割（中文句号、换行等）
    sentences = re.split(r'(?<=[。！？；\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1 or embed_model is None:
        # 退化为段落滑动窗口
        return _fallback_split(text, chunk_size, chunk_overlap)

    # 计算每个句子的嵌入
    try:
        # 兼容多种 embedding 后端
        if hasattr(embed_model, 'encode'):
            output = embed_model.encode(sentences)
            if isinstance(output, dict):
                embeddings = output['dense_vecs']
            else:
                embeddings = np.array(output)
        else:
            embeddings = embed_model.get_text_embedding_batch(sentences)
            embeddings = np.array(embeddings)
    except Exception:
        return _fallback_split(text, chunk_size, chunk_overlap)

    # 计算相邻句子相似度
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
        )
        similarities.append(float(sim))

    # 在相似度低谷处切分
    chunks = []
    current = sentences[0]
    for i in range(1, len(sentences)):
        if len(current) >= chunk_size:
            chunks.append(current)
            # 重叠：保留最后一句
            current = sentences[i - 1] + " " + sentences[i] if i > 1 else sentences[i]
        elif similarities[i - 1] < similarity_threshold and len(current) >= 100:
            chunks.append(current)
            current = sentences[i]
        else:
            current += " " + sentences[i]
    if current.strip():
        chunks.append(current)

    # 合并过短的块
    chunks = _merge_short(chunks)
    logger.info("语义分块: %d 句 → %d 块 (阈值=%.2f)", len(sentences), len(chunks), similarity_threshold)
    return chunks


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """段落感知滑动窗口分块。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunk = para[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += chunk_size - chunk_overlap
                if start >= len(para):
                    break
    return _merge_short(chunks)


def _merge_short(chunks: List[str], min_len: int = 80) -> List[str]:
    """合并过短片段。"""
    if len(chunks) <= 1:
        return chunks
    merged, buf, buf_len = [], "", 0
    for c in chunks:
        if buf_len > 0 and buf_len + len(c) < min_len * 3:
            buf += "\n" + c
            buf_len += len(c)
        elif len(c) < min_len:
            if buf:
                merged.append(buf)
            buf, buf_len = c, len(c)
        else:
            if buf:
                merged.append(buf)
                buf, buf_len = "", 0
            merged.append(c)
    if buf:
        if merged and len(buf) < min_len:
            merged[-1] += "\n" + buf
        else:
            merged.append(buf)
    return merged


# ======== Small-to-Big 检索支持 ========

def make_small_chunks(chunks: List[str], small_size: int = 200) -> List[Tuple[str, int]]:
    """为 Small-to-Big 策略创建小粒度索引块。

    返回: [(small_chunk_text, parent_chunk_index), ...]
    检索时用小块匹配，返回时返回完整的父块上下文。
    """
    small_chunks = []
    for parent_idx, chunk in enumerate(chunks):
        if len(chunk) <= small_size:
            small_chunks.append((chunk, parent_idx))
        else:
            sentences = re.split(r'(?<=[。！？；])\s*', chunk)
            current = ""
            for s in sentences:
                if len(current) + len(s) > small_size and current:
                    small_chunks.append((current.strip(), parent_idx))
                    current = s
                else:
                    current += s
            if current.strip():
                small_chunks.append((current.strip(), parent_idx))
    return small_chunks


def expand_to_big_context(
    small_hits: List[dict],
    parent_chunks: List[str],
    neighbor_window: int = 2,
) -> List[str]:
    """将小块检索结果展开为大块上下文。

    对每个命中的小块，获取其父块及相邻块。
    """
    seen_indices = set()
    contexts = []
    for hit in small_hits:
        parent_idx = hit.get("metadata", {}).get("parent_chunk_index", 0)
        for offset in range(-neighbor_window, neighbor_window + 1):
            idx = parent_idx + offset
            if 0 <= idx < len(parent_chunks) and idx not in seen_indices:
                seen_indices.add(idx)
                contexts.append(parent_chunks[idx])
    return contexts

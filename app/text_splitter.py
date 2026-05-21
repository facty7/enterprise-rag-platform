"""
文本切分 —— 段落感知的滑动窗口策略，支持页码追踪
"""
import logging
import re
from typing import List, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


def split_text_into_chunks(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """向后兼容的简单切分（不含页码信息）。"""
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

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

    chunks = _merge_small_chunks(chunks)
    logger.info("文本切分: %d 字符 → %d 个块", len(text), len(chunks))
    return chunks


def split_text_into_chunks_with_pages(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Tuple[str, int]]:
    """
    切分文本并追踪每个块所属页码。
    文本中的 [第N页] 标记会被识别并关联到该段落的所有块。
    返回: [(chunk_text, page_number), ...]
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    if not text or not text.strip():
        return []

    # 按段落分割
    raw_paragraphs = text.split("\n\n")
    paragraphs = []  # [(text, page_num), ...]

    current_page = 1
    page_pattern = re.compile(r'\[第(\d+)页\]\s*')

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # 检测页码标记
        m = page_pattern.match(para)
        if m:
            current_page = int(m.group(1))
            para = page_pattern.sub('', para, count=1).strip()
            if not para:
                continue

        if len(para) <= chunk_size:
            paragraphs.append((para, current_page))
        else:
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunk = para[start:end].strip()
                if chunk:
                    paragraphs.append((chunk, current_page))
                start += chunk_size - chunk_overlap
                if start >= len(para):
                    break

    result = [(text, page) for text, page in paragraphs]
    result = _merge_small_chunks_with_pages(result)
    logger.info("文本切分（含页码）: %d 字符 → %d 个块", len(text), len(result))
    return result


def _merge_small_chunks(chunks: List[str], min_len: int = 80) -> List[str]:
    """合并相邻短片段，减少碎片化。"""
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


def _merge_small_chunks_with_pages(cwp: List[Tuple[str, int]], min_len: int = 80) -> List[Tuple[str, int]]:
    """合并相邻短片段（保留页码）。"""
    if len(cwp) <= 1:
        return cwp
    merged, buf, buf_len, buf_page = [], "", 0, 0
    for text, page in cwp:
        if buf_len > 0 and buf_len + len(text) < min_len * 3:
            buf += "\n" + text
            buf_len += len(text)
        elif len(text) < min_len:
            if buf:
                merged.append((buf, buf_page))
            buf, buf_len, buf_page = text, len(text), page
        else:
            if buf:
                merged.append((buf, buf_page))
                buf, buf_len = "", 0
            merged.append((text, page))
    if buf:
        if merged and len(buf) < min_len:
            merged[-1] = (merged[-1][0] + "\n" + buf, merged[-1][1])
        else:
            merged.append((buf, buf_page))
    return merged

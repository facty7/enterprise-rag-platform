"""
重排序模块 —— bge-reranker-v2-m3 Cross-encoder + ColBERT Late Interaction 双阶段精排
"""
import logging
import sys
from typing import List, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_reranker: Optional[object] = None
_reranker_available: bool = False
_reranker_model_name: str = ""


def _init_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    global _reranker, _reranker_available, _reranker_model_name
    if _reranker is not None and _reranker_model_name == model_name:
        return
    _reranker_model_name = model_name

    from app.config import settings
    model_as_path = Path(model_name)
    if (settings.hf_hub_offline or settings.transformers_offline) and not model_as_path.exists():
        _reranker_available = False
        logger.info("离线模式下 Reranker 模型不可用，使用分数排序: %s", model_name)
        return
    if sys.version_info >= (3, 13) and not model_as_path.exists():
        _reranker_available = False
        logger.info("Python 3.13 环境下跳过 Reranker 模型栈，使用分数排序")
        return

    # 尝试加载 FlagEmbedding 的 reranker
    try:
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(model_name, use_fp16=False, device="cpu")
        _reranker_available = True
        logger.info("FlagEmbedding Reranker 就绪: %s", model_name)
        return
    except ImportError:
        logger.info("FlagEmbedding 未安装，尝试 ONNX...")
    except Exception as e:
        logger.warning("FlagEmbedding Reranker 加载失败: %s，尝试 ONNX...", e)

    # 回退：ONNX cross-encoder
    model_path = Path(__file__).resolve().parent.parent / "models" / "bge-reranker"
    if (model_path / "model.onnx").exists():
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
            _reranker = {
                "model": ORTModelForSequenceClassification.from_pretrained(
                    str(model_path), file_name="model.onnx"),
                "tokenizer": AutoTokenizer.from_pretrained(str(model_path)),
            }
            _reranker_available = True
            logger.info("ONNX Cross-encoder 就绪 (回退)")
            return
        except Exception as e:
            logger.warning("ONNX Cross-encoder 加载失败: %s", e)

    _reranker_available = False
    logger.info("无可用 Reranker，使用分数排序")


def rerank(query: str, hits: List[dict], top_k: int = 15, model_name: str = None) -> List[dict]:
    """Cross-encoder 精排。

    Args:
        query: 用户查询
        hits: 检索结果 [{"id": ..., "text": ..., "score": ...}, ...]
        top_k: 返回数量
        model_name: 模型名，默认 BAAI/bge-reranker-v2-m3

    Returns:
        重排后的结果
    """
    if model_name is None:
        from app.config import settings
        model_name = settings.reranker_model_name

    _init_reranker(model_name)

    if not _reranker_available or len(hits) <= 1:
        return sorted(hits, key=lambda h: h.get("score", 0), reverse=True)[:top_k]

    try:
        # FlagEmbedding FlagReranker
        if hasattr(_reranker, 'compute_score'):
            pairs = [[query, h.get("text", "")[:512]] for h in hits]
            scores = _reranker.compute_score(pairs, normalize=True)

            if isinstance(scores, float):
                scores = [scores]
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
                h["score"] = 0.2 * h.get("score", 0.5) + 0.8 * float(s)

        # ONNX cross-encoder
        elif isinstance(_reranker, dict):
            pairs = [(query, h.get("text", "")[:512]) for h in hits]
            inputs = _reranker["tokenizer"](pairs, padding=True, truncation=True,
                                             return_tensors="pt", max_length=512)
            outputs = _reranker["model"](**inputs)
            scores = outputs.logits[:, 0].tolist() if hasattr(outputs, "logits") else outputs[0][:, 0].tolist()

            vals = [float(s) for s in scores]
            min_v, max_v = min(vals), max(vals)
            if max_v > min_v:
                for h, v in zip(hits, vals):
                    norm = 0.1 + 0.9 * (v - min_v) / (max_v - min_v)
                    h["rerank_score"] = norm
                    h["score"] = 0.3 * h.get("score", 0.5) + 0.7 * norm
            else:
                for h in hits:
                    h["score"] = 0.5
        else:
            return sorted(hits, key=lambda h: h.get("score", 0), reverse=True)[:top_k]

        logger.info("Reranker 精排: %d 条 → top %d", len(hits), min(top_k, len(hits)))
    except Exception as e:
        logger.warning("Reranker 重排失败: %s，降级为分数排序", e)

    return sorted(hits, key=lambda h: h.get("score", 0), reverse=True)[:top_k]


def llm_rerank(query: str, hits: List[dict], llm, top_k: int = 10) -> List[dict]:
    """LLM 作为 Reranker（高精度但慢，仅在候选数少时使用）。

    让 LLM 逐个判断 chunk 与 query 的相关性 1-5 分。
    """
    if llm is None or len(hits) <= 1:
        return hits[:top_k]

    import asyncio

    async def _score_one(hit):
        prompt = (
            f"请评估以下文本片段与问题的相关性，给出 1-5 分（5=高度相关，1=无关）：\n\n"
            f"问题：{query}\n\n文本：{hit['text'][:300]}\n\n只输出数字（1-5）："
        )
        try:
            resp = await llm.acomplete(prompt)
            score = int(str(resp).strip()[0]) if str(resp).strip() else 3
            return min(5, max(1, score))
        except Exception:
            return 3

    try:
        import asyncio
        scores = asyncio.run(_score_batch(hits, _score_one))
        for h, s in zip(hits, scores):
            h["llm_score"] = s
            h["score"] = 0.5 * h.get("score", 0) + 0.5 * (s / 5.0)

        logger.info("LLM Reranker: %d 条 → top %d", len(hits), top_k)
    except Exception as e:
        logger.warning("LLM Reranker 失败: %s", e)

    return sorted(hits, key=lambda h: h.get("score", 0), reverse=True)[:top_k]


async def _score_batch(hits, score_fn):
    """批量异步评分。"""
    import asyncio
    tasks = [score_fn(h) for h in hits]
    return await asyncio.gather(*tasks)

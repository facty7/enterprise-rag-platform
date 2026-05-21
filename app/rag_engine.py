"""
Enterprise RAG engine —— hybrid retrieval pipeline

Pipeline:
  1. Query 处理 (LLM改写 + HyDE + Multi-Query + 分解)
  2. 三路并行检索 (Dense + BGE-M3 Sparse + BM25倒排)
  3. RRF 融合
  4. ColBERT Late Interaction 精排
  5. Self-RAG 质量自评 (不够 → 重检索)
  6. Cross-encoder Reranker (bge-reranker-v2-m3)
  7. GraphRAG 实体增强
  8. MMR 多样性 + 文档分组
"""
import logging
import uuid
import math
import json
import time
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings, get_search_mode_profile
from app.retrieval import (
    InMemorySparseIndex, reciprocal_rank_fusion,
    BGE_M3_SparseRetriever, ColBERTLateInteraction,
)
from app.query_processor import is_aggregate_query

logger = logging.getLogger(__name__)

_EMBED_DIM: Optional[int] = None
_EMBED_MODEL = None


class HashingEmbedding:
    """Tiny local embedding fallback so a fresh clone can start without model files."""

    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
            analyzer="char_wb",
            ngram_range=(2, 4),
        )

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        return matrix.astype(np.float32).toarray().tolist()

    def get_text_embedding(self, text: str) -> list[float]:
        return self.get_text_embedding_batch([text])[0]


def _use_hashing_embedding(reason: str):
    """Switch to a deterministic local fallback without importing heavy model stacks."""
    global _EMBED_MODEL, _EMBED_DIM
    logger.warning("%s，启用本地 HashingEmbedding 演示模式", reason)
    _EMBED_MODEL = HashingEmbedding()
    _EMBED_DIM = len(_EMBED_MODEL.get_text_embedding("probe"))
    logger.info("HashingEmbedding 就绪: %dd (demo fallback)", _EMBED_DIM)
    return _EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():
    """智能加载嵌入模型 —— 自动选择最佳可用后端。

    优先级:
    1. BGE-M3 via FlagEmbedding (dense+sparse+colbert)
    2. BGE-base via sentence-transformers (dense only)
    3. Fallback to HuggingFaceEmbedding
    """
    global _EMBED_MODEL, _EMBED_DIM
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    model_name = settings.embed_model_name
    is_bge_m3 = "m3" in model_name.lower()

    logger.info("加载嵌入模型: %s", model_name)

    model_path = Path(model_name)
    if (settings.hf_hub_offline or settings.transformers_offline) and not model_path.exists():
        return _use_hashing_embedding(f"离线模式下模型不可用: {model_name}")
    if sys.version_info >= (3, 13) and not model_path.exists():
        return _use_hashing_embedding("Python 3.13 环境下跳过 PyTorch 模型栈以保证可启动")

    if is_bge_m3:
        # 尝试 FlagEmbedding BGEM3FlagModel (dense + sparse + colbert)
        try:
            from FlagEmbedding import BGEM3FlagModel
            _EMBED_MODEL = BGEM3FlagModel(
                model_name, use_fp16=settings.embed_use_fp16,
                device=settings.embed_device,
            )
            probe = _EMBED_MODEL.encode(["probe"], return_dense=True)['dense_vecs']
            _EMBED_DIM = probe.shape[1]
            logger.info("BGE-M3 ready: dense=%dd, sparse=Y, colbert=Y", _EMBED_DIM)
            return _EMBED_MODEL
        except Exception as e:
            logger.warning("BGE-M3 via FlagEmbedding 失败: %s，回退 sentence-transformers", e)

    # 通用路径: sentence-transformers (dense only, 兼容 BGE-base/BGE-large/BGE-M3)
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(model_name, device=settings.embed_device)
        _EMBED_DIM = _EMBED_MODEL.get_sentence_embedding_dimension()
        logger.info("SentenceTransformer 就绪: %dd", _EMBED_DIM)
        return _EMBED_MODEL
    except Exception as e:
        logger.warning("SentenceTransformer 失败: %s，回退 HuggingFace", e)

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=model_name, device=settings.embed_device,
        )
        _EMBED_DIM = len(_EMBED_MODEL.get_text_embedding("probe"))
        logger.info("HuggingFace Embedding 就绪: %dd", _EMBED_DIM)
        return _EMBED_MODEL
    except Exception as e:
        logger.warning("HuggingFace Embedding 失败: %s，启用本地 HashingEmbedding 演示模式", e)

    return _use_hashing_embedding("未找到可用嵌入模型")


@lru_cache(maxsize=1)
def get_llm():
    """获取 LLM 实例 (DeepSeek)。"""
    from llama_index.llms.openai_like import OpenAILike
    key = settings.get_llm_api_key().strip()
    if not key or len(key) < 20:
        logger.warning("LLM_API_KEY 未配置")
        return None
    return OpenAILike(
        model=settings.llm_model_name,
        api_base=settings.llm_api_base,
        api_key=key,
        temperature=0.0,
        max_tokens=4096,
        is_chat_model=True,
    )


def _has_flag_embedding() -> bool:
    """检查模型是否支持 BGE-M3 的全部特性（sparse/ColBERT）。"""
    emb = get_embedding_model()
    cls_name = type(emb).__name__
    # FlagEmbedding 的 BGEM3FlagModel 实际类名为 M3Embedder
    if 'M3' in cls_name or 'BGEM3' in cls_name or 'Flag' in cls_name:
        return True
    # SentenceTransformer 不支持 sparse/colbert
    if 'SentenceTransformer' in cls_name or 'Transformer' in cls_name:
        return False
    # 尝试检测 encode(return_sparse=True) 是否可用
    try:
        o = emb.encode(['test'], return_sparse=True)
        return 'lexical_weights' in o
    except Exception:
        return False


def _encode_texts(embed, texts: list) -> np.ndarray:
    """统一编码接口：兼容 FlagEmbedding/SentenceTransformer/HuggingFace。"""
    # FlagEmbedding (BGEM3FlagModel / FlagModel)
    if hasattr(embed, 'encode') and hasattr(embed, 'model'):
        output = embed.encode(texts, return_dense=True)
        if isinstance(output, dict):
            return np.array(output['dense_vecs'])
        return np.array(output)
    # SentenceTransformer
    if hasattr(embed, 'encode'):
        return np.array(embed.encode(texts))
    # HuggingFace Embedding (llama-index)
    if hasattr(embed, 'get_text_embedding_batch'):
        return np.array(embed.get_text_embedding_batch(texts))
    # fallback
    return np.array([embed.get_text_embedding(t) for t in texts])


def _encode_query(embed, query: str) -> list:
    """统一查询编码接口。"""
    vecs = _encode_texts(embed, [query])
    return vecs[0].tolist()


_rag_instance: Optional["EnterpriseRAGEngine"] = None


def get_rag_engine() -> "EnterpriseRAGEngine":
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = EnterpriseRAGEngine()
    return _rag_instance


class EnterpriseRAGEngine:
    """企业 RAG 引擎 —— 三路检索 + ColBERT + Self-RAG + GraphRAG。"""

    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._shared = settings.shared_collection
        self._internal = settings.internal_collection
        self._bm25_index: Optional[InMemorySparseIndex] = None
        self._sparse_retriever: Optional[BGE_M3_SparseRetriever] = None
        self._colbert: Optional[ColBERTLateInteraction] = None
        self._entity_graph = None
        self._graph_loaded = False

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=settings.qdrant_path)
            for col in [self._shared, self._internal]:
                self._ensure_collection(col)
        return self._client

    @property
    def bm25(self) -> InMemorySparseIndex:
        if self._bm25_index is None:
            self._bm25_index = InMemorySparseIndex()
        return self._bm25_index

    @property
    def sparse_retriever(self) -> Optional[BGE_M3_SparseRetriever]:
        if self._sparse_retriever is None and _has_flag_embedding():
            self._sparse_retriever = BGE_M3_SparseRetriever(
                self.client, get_embedding_model()
            )
        return self._sparse_retriever

    @property
    def colbert(self) -> Optional[ColBERTLateInteraction]:
        if self._colbert is None and _has_flag_embedding():
            self._colbert = ColBERTLateInteraction(get_embedding_model())
        return self._colbert

    @property
    def entity_graph(self):
        if not self._graph_loaded:
            from app.graph_rag import EntityGraph
            self._entity_graph = EntityGraph()
            graph_path = settings.qdrant_path + "/entity_graph.json"
            import os
            if os.path.exists(graph_path):
                self._entity_graph.load(graph_path)
            self._graph_loaded = True
        return self._entity_graph

    def close(self):
        """Release local vector-store resources explicitly on shutdown."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                logger.debug("Qdrant close skipped: %s", exc)
            finally:
                self._client = None

    def _ensure_collection(self, name: str):
        """确保 Qdrant 集合存在。"""
        try:
            self._client.get_collection(name)
            return
        except (UnexpectedResponse, ValueError):
            dim = self._get_embed_dim()
            self._client.create_collection(
                name,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
            logger.info("创建集合: %s (%dd)", name, dim)

    def _ensure_bm25_index(self):
        """从 Qdrant 延迟构建 BM25 倒排索引。"""
        if self._bm25_index is not None and self._bm25_index._built:
            return
        logger.info("构建 BM25 倒排索引...")
        idx = self.bm25
        idx.reset()
        items = []
        for col in [self._shared, self._internal]:
            offset = None
            while True:
                page, offset = self.client.scroll(
                    collection_name=col, limit=1000,
                    with_payload=["text"], with_vectors=False, offset=offset,
                )
                for pt in page:
                    items.append((pt.id, pt.payload.get("text", "")))
                if offset is None:
                    break
        for pid, text in items:
            idx.index_point(pid, text)
        if idx.doc_lengths:
            idx.avg_doc_len = sum(idx.doc_lengths.values()) / len(idx.doc_lengths)
        idx._built = True
        logger.info("BM25 倒排索引就绪: %d 文档", idx.total_docs)

    # ============ 索引 ============

    def index_document(self, chunks: List[str], metadata: Optional[dict] = None,
                       per_chunk_meta: Optional[List[dict]] = None,
                       collection: str = None,
                       display_texts: List[str] = None) -> int:
        if not chunks:
            return 0
        target = collection or settings.shared_collection
        embed = get_embedding_model()

        # 向量化：dense + sparse（如果可用）
        dense_vectors = _encode_texts(embed, chunks)

        points = []
        for i, chunk in enumerate(chunks):
            pid = str(uuid.uuid4())
            display = display_texts[i] if display_texts and i < len(display_texts) else chunk
            payload = {"text": chunk, "display_text": display, "chunk_index": i}
            if metadata:
                payload.update(metadata)
            if per_chunk_meta and i < len(per_chunk_meta) and per_chunk_meta[i]:
                payload.update(per_chunk_meta[i])

            points.append(qm.PointStruct(
                id=pid, vector=dense_vectors[i].tolist(), payload=payload
            ))

            # 同步更新 BM25 倒排索引
            if self._bm25_index is not None and self._bm25_index._built:
                self._bm25_index.index_point(pid, chunk)

        self.client.upsert(collection_name=target, points=points, wait=True)
        logger.info("已入库 %d 条到 %s (dense%s)", len(points), target,
                     "+sparse" if _has_flag_embedding() else "")
        return len(points)

    def delete_document(self, file_hash: str) -> bool:
        deleted = False
        for col in [self._shared, self._internal]:
            try:
                ids, offset = [], None
                while True:
                    page, offset = self.client.scroll(
                        collection_name=col, limit=1000,
                        with_payload=["file_hash"], with_vectors=False, offset=offset)
                    for pt in page:
                        if pt.payload and pt.payload.get("file_hash") == file_hash:
                            ids.append(pt.id)
                    if offset is None:
                        break
                if ids:
                    self.client.delete(collection_name=col,
                                       points_selector=qm.PointIdsList(points=ids))
                    if self._bm25_index is not None:
                        for pid in ids:
                            self._bm25_index.remove_point(pid)
                    logger.info("已删除 %d 条", len(ids))
                    deleted = True
            except Exception as e:
                logger.warning("删除异常: %s", e)
        return deleted

    def delete_by_filename(self, file_name: str, collection: str = None) -> int:
        deleted = 0
        cols = [collection] if collection else [self._shared, self._internal]
        for col in cols:
            try:
                ids, offset = [], None
                while True:
                    page, offset = self.client.scroll(
                        collection_name=col, limit=1000,
                        with_payload=["file_name"], with_vectors=False, offset=offset)
                    for pt in page:
                        if pt.payload and pt.payload.get("file_name") == file_name:
                            ids.append(pt.id)
                    if offset is None:
                        break
                if ids:
                    self.client.delete(collection_name=col,
                                       points_selector=qm.PointIdsList(points=ids))
                    if self._bm25_index is not None:
                        for pid in ids:
                            self._bm25_index.remove_point(pid)
                    deleted += len(ids)
                    logger.info("覆盖模式: 已删除 %s 的 %d 条", file_name, len(ids))
            except Exception as e:
                logger.warning("delete_by_filename异常 [%s]: %s", col, e)
        return deleted

    # ============ 检索主流程 ============

    async def retrieve(self, query: str, top_k: int = None,
                       user_can_see: List[str] = None,
                       use_hyde: bool = None,
                       use_multi_query: bool = None,
                       use_late_interaction: bool = None,
                       use_self_rag: bool = None,
                       use_graph_rag: bool = None,
                       strategy_mode: str = None,
                       return_trace: bool = False):
        """检索主入口 —— 全链路优化。

        检索流程：
        1. Query 优化：LLM 改写 + HyDE + Multi-Query
        2. 三路检索：Dense + Sparse(BGE-M3) + BM25
        3. RRF 融合
        4. ColBERT Late Interaction
        5. Self-RAG 质量评估 & 重检索
        6. Cross-encoder Reranker
        7. GraphRAG 实体增强
        8. MMR 多样性
        """
        mode = get_search_mode_profile(strategy_mode)
        if top_k is None:
            top_k = settings.top_k_retrieval
        if use_hyde is None:
            use_hyde = mode["hyde"] and settings.hyde_enabled
        if use_multi_query is None:
            use_multi_query = mode["multi_query"] and settings.multi_query_enabled
        if use_late_interaction is None:
            use_late_interaction = mode["late_interaction"] and settings.late_interaction_enabled
        if use_self_rag is None:
            use_self_rag = mode["self_rag"] and settings.self_rag_enabled
        if use_graph_rag is None:
            use_graph_rag = mode["graph_rag"] and settings.graph_rag_enabled

        use_query_rewrite = bool(mode["query_rewrite"])
        use_query_decompose = bool(mode["query_decompose"])
        query_variants_limit = int(mode["query_variants_limit"])
        candidate_count = int(mode["retrieval_candidates"])

        timings: Dict[str, float] = {}
        trace: Dict[str, object] = {
            "mode": mode["name"],
            "mode_label": mode["label"],
            "mode_description": mode["description"],
            "requested_mode": strategy_mode or settings.search_mode_default,
            "actual_mode": mode["name"],
            "actual_mode_label": mode["label"],
            "actual_mode_description": mode["description"],
            "query": query,
            "flags": {
                "query_rewrite": use_query_rewrite,
                "hyde": use_hyde,
                "multi_query": use_multi_query,
                "query_decompose": use_query_decompose,
                "late_interaction": use_late_interaction,
                "self_rag": use_self_rag,
                "graph_rag": use_graph_rag,
            },
            "timing_ms": timings,
            "phase": {},
            "query_variants": [],
            "candidate_count": candidate_count,
            "final_count": 0,
        }
        t_total = time.perf_counter()

        t_index = time.perf_counter()
        self._ensure_bm25_index()
        timings["bm25_index_ms"] = (time.perf_counter() - t_index) * 1000.0
        llm = get_llm()

        # ====== Phase 1: Query 优化（按模式启用）======
        search_queries = [query]
        aggregate_like = is_aggregate_query(query) or len(query) > 50

        if llm:
            import asyncio
            from app.query_processor import (
                llm_query_rewrite, generate_hyde, generate_multi_queries,
                decompose_query,
            )

            task_names = []
            task_objs = []
            if use_query_rewrite:
                task_names.append("rewrite")
                task_objs.append(llm_query_rewrite(query, llm))
            if use_hyde:
                task_names.append("hyde")
                task_objs.append(generate_hyde(query, llm))
            if use_multi_query:
                task_names.append("multi")
                task_objs.append(generate_multi_queries(query, llm, settings.multi_query_count))
            if use_query_decompose and aggregate_like:
                task_names.append("decompose")
                task_objs.append(decompose_query(query, llm))

            if task_objs:
                t_phase = time.perf_counter()
                results = await asyncio.gather(*task_objs, return_exceptions=True)
                timings["query_opt_ms"] = (time.perf_counter() - t_phase) * 1000.0
                llm_map = dict(zip(task_names, results))

                rewritten = llm_map.get("rewrite")
                if isinstance(rewritten, str):
                    rewritten = rewritten.strip()
                    if rewritten and rewritten != query:
                        search_queries.insert(0, rewritten)

                hyde_text = llm_map.get("hyde")
                if isinstance(hyde_text, str):
                    hyde_text = hyde_text.strip()
                    if hyde_text:
                        search_queries.append(hyde_text)

                variants = llm_map.get("multi")
                if isinstance(variants, list):
                    search_queries.extend([
                        q.strip() for q in variants
                        if isinstance(q, str) and q.strip()
                    ])

                decomposed = llm_map.get("decompose")
                if isinstance(decomposed, list):
                    search_queries.extend([
                        q.strip() for q in decomposed
                        if isinstance(q, str) and q.strip() and q.strip() != query
                    ])

        # 去重
        seen_q = set()
        unique_queries = []
        for q in search_queries:
            q = q.strip()
            if q and q not in seen_q:
                seen_q.add(q)
                unique_queries.append(q)

        query_limit = max(1, query_variants_limit)
        active_queries = unique_queries[:query_limit]
        trace["query_variants"] = active_queries[:4]
        trace["query_variants_count"] = len(unique_queries)

        logger.info("Query variants (%s): %d -> %d", mode["name"], len(search_queries), len(active_queries))

        # ====== Phase 2: 三路并行检索 ======
        t_phase = time.perf_counter()
        all_fused = {}  # pid → score (best across all query variants)
        embed = get_embedding_model()

        for sq in active_queries:
            dense_results = []
            sparse_results = []
            keyword_results = []
            dense_seen = set()
            sparse_seen = set()
            keyword_seen = set()

            # 2a. Dense 检索 (Qdrant ANN)
            t_dense = time.perf_counter()
            qv = _encode_query(embed, sq)
            for col_name in [self._shared, self._internal]:
                results = self.client.query_points(
                    collection_name=col_name, query=qv, limit=candidate_count,
                    with_payload=True, with_vectors=False,
                )
                for pt in results.points:
                    if pt.id in dense_seen:
                        continue
                    if not self._check_access(col_name, pt, user_can_see):
                        continue
                    dense_seen.add(pt.id)
                    dense_results.append((pt.id, pt.score))
            timings["dense_ms"] = timings.get("dense_ms", 0.0) + (time.perf_counter() - t_dense) * 1000.0

            # 2b. BGE-M3 Sparse 检索
            if self.sparse_retriever is not None:
                t_sparse = time.perf_counter()
                for col_name in [self._shared, self._internal]:
                    sr = self.sparse_retriever.search(sq, col_name, top_k=candidate_count)
                    for pid, score in sr:
                        if pid in sparse_seen:
                            continue
                        sparse_seen.add(pid)
                        sparse_results.append((pid, score))
                timings["sparse_ms"] = timings.get("sparse_ms", 0.0) + (time.perf_counter() - t_sparse) * 1000.0

            # 2c. BM25 倒排索引关键词检索
            t_bm25 = time.perf_counter()
            bm25_raw = self.bm25.search(sq, top_k=candidate_count)
            best_bm25 = bm25_raw[0][1] if bm25_raw else 1.0
            for pid, score in bm25_raw:
                if pid in keyword_seen:
                    continue
                keyword_seen.add(pid)
                keyword_results.append((pid, score / max(best_bm25, 1e-6)))
            timings["bm25_ms"] = timings.get("bm25_ms", 0.0) + (time.perf_counter() - t_bm25) * 1000.0

            # 2d. RRF 三路融合
            t_rrf = time.perf_counter()
            fused = reciprocal_rank_fusion(
                dense_results,
                sparse_results if sparse_results else None,
                keyword_results if keyword_results else None,
                k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
                keyword_weight=settings.keyword_weight,
                top_k=candidate_count,
            )
            timings["rrf_ms"] = timings.get("rrf_ms", 0.0) + (time.perf_counter() - t_rrf) * 1000.0

            # 合并到全局分数（取最大值）
            for pid, score in fused:
                if pid not in all_fused or score > all_fused[pid]:
                    all_fused[pid] = score

        # 排序并取候选
        ranked_pids = sorted(all_fused.items(), key=lambda x: x[1], reverse=True)
        trace["ranked_candidates"] = len(ranked_pids)
        logger.info("三路融合: %d 个候选 (来自 %d 个查询变体)", len(ranked_pids), len(active_queries))

        # ====== Phase 3: 加载候选片段详情 ======
        t_phase = time.perf_counter()
        candidate_chunks = self._load_chunks_by_ids([pid for pid, _ in ranked_pids[:candidate_count * 2]])
        timings["load_chunks_ms"] = (time.perf_counter() - t_phase) * 1000.0
        trace["loaded_chunks"] = len(candidate_chunks)

        # ====== Phase 4: ColBERT Late Interaction ======
        if use_late_interaction and self.colbert is not None and len(candidate_chunks) > 5:
            t_phase = time.perf_counter()
            cand_for_colbert = [
                (c["id"], c["text"], c.get("score", 0))
                for c in candidate_chunks[:settings.colbert_top_k]
            ]
            colbert_scores = self.colbert.rerank(query, cand_for_colbert, top_k=min(30, len(cand_for_colbert)))
            colbert_map = dict(colbert_scores)
            for c in candidate_chunks:
                if c["id"] in colbert_map:
                    c["score"] = 0.4 * c.get("score", 0) + 0.6 * colbert_map[c["id"]]
            candidate_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
            timings["colbert_ms"] = (time.perf_counter() - t_phase) * 1000.0
            trace["colbert_used"] = True
            logger.info("ColBERT Late Interaction: %d 条重排", len(colbert_scores))
        else:
            trace["colbert_used"] = False

        # ====== Phase 5: Cross-encoder Reranker ======
        if settings.reranker_enabled and len(candidate_chunks) > 5:
            t_phase = time.perf_counter()
            from app.reranker import rerank
            candidate_chunks = rerank(query, candidate_chunks, top_k=top_k * 2)
            timings["reranker_ms"] = (time.perf_counter() - t_phase) * 1000.0

        # ====== Phase 6: Self-RAG —— 仅复杂查询触发 ======
        is_complex = aggregate_like
        if use_self_rag and is_complex and llm is not None and len(candidate_chunks) >= 10:
            t_phase = time.perf_counter()
            candidate_chunks = await self._self_rag_evaluate(
                query, candidate_chunks, llm, user_can_see
            )
            timings["self_rag_ms"] = (time.perf_counter() - t_phase) * 1000.0
            trace["self_rag_triggered"] = True
        else:
            trace["self_rag_triggered"] = False

        # ====== Phase 7: 日期加权 ======
        from datetime import datetime, timezone
        now_dt = datetime.now(timezone.utc)
        t_phase = time.perf_counter()
        for c in candidate_chunks:
            rd = c.get("metadata", {}).get("report_date", "")
            if not rd:
                continue
            try:
                d = datetime.strptime(rd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                delta = (now_dt - d).days
                if delta <= 1:
                    c["score"] *= 1.2
                elif delta <= 7:
                    c["score"] *= 1.1
            except ValueError:
                pass
        candidate_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        timings["date_boost_ms"] = (time.perf_counter() - t_phase) * 1000.0

        # ====== Phase 8: 聚合查询窗口扩展 ======
        t_phase = time.perf_counter()
        if aggregate_like:
            candidate_chunks = self._expand_window(candidate_chunks, user_can_see)
        timings["window_ms"] = (time.perf_counter() - t_phase) * 1000.0

        final = candidate_chunks[:top_k]
        trace["final_count"] = len(final)

        # ====== Phase 9: GraphRAG 实体增强（附加到结果 metadata） ======
        if use_graph_rag:
            t_phase = time.perf_counter()
            graph_ctx = self.entity_graph.build_context_for_query(query)
            if graph_ctx and final:
                # 将图谱上下文附加到第一个结果中，prompt_builder 会使用
                final[0]["graph_context"] = graph_ctx
                trace["graph_context_used"] = True
            else:
                trace["graph_context_used"] = False
            timings["graph_ms"] = (time.perf_counter() - t_phase) * 1000.0
        else:
            trace["graph_context_used"] = False

        timings["retrieval_ms"] = (
            timings.get("dense_ms", 0.0)
            + timings.get("sparse_ms", 0.0)
            + timings.get("bm25_ms", 0.0)
            + timings.get("rrf_ms", 0.0)
        )
        trace["timing_ms"] = {k: round(v, 1) for k, v in timings.items()}
        trace["phase"] = {
            "query_opt": round(timings.get("query_opt_ms", 0.0), 1),
            "retrieval": round(timings.get("retrieval_ms", 0.0), 1),
            "load_chunks": round(timings.get("load_chunks_ms", 0.0), 1),
            "colbert": round(timings.get("colbert_ms", 0.0), 1),
            "reranker": round(timings.get("reranker_ms", 0.0), 1),
            "self_rag": round(timings.get("self_rag_ms", 0.0), 1),
            "window": round(timings.get("window_ms", 0.0), 1),
            "graph": round(timings.get("graph_ms", 0.0), 1),
            "date_boost": round(timings.get("date_boost_ms", 0.0), 1),
        }
        trace["total_ms"] = round((time.perf_counter() - t_total) * 1000.0, 1)

        if return_trace:
            return final, trace
        return final

    # ============ Self-RAG ============

    async def _self_rag_evaluate(self, query: str, chunks: List[dict], llm,
                                  user_can_see: List[str] = None) -> List[dict]:
        """Self-RAG: LLM 评估检索质量，不够则触发重检索。"""
        from app.query_processor import extract_keywords

        for round_num in range(settings.self_rag_max_rounds):
            # 取 Top-5 让 LLM 评估相关性
            top_sample = chunks[:5]
            eval_prompt = (
                f"评估以下检索结果与问题的整体相关性（0-10分，10=完美匹配）：\n\n"
                f"问题：{query}\n\n"
                f"检索片段：\n" +
                "\n".join(f"[{i}] {c['text'][:200]}" for i, c in enumerate(top_sample)) +
                "\n\n只输出数字（0-10）："
            )

            try:
                resp = await llm.acomplete(eval_prompt)
                score = float(str(resp).strip())
                score = max(0, min(10, score))
            except Exception:
                score = 7.0  # 默认通过

            if score >= settings.self_rag_threshold * 10:
                logger.info("Self-RAG: 第%d轮质量评分 %.1f ≥ %.1f，通过",
                           round_num + 1, score, settings.self_rag_threshold * 10)
                break

            logger.info("Self-RAG: 第%d轮质量评分 %.1f < %.1f，触发重检索",
                       round_num + 1, score, settings.self_rag_threshold * 10)

            # 重检索：用完全不同的查询策略（只用关键词）
            keywords = extract_keywords(query, top_n=5)
            kw_query = " ".join(keywords)
            bm25_results = self.bm25.search(kw_query, top_k=30)
            new_chunks = self._load_chunks_by_ids([pid for pid, _ in bm25_results])
            if new_chunks:
                # 合并新旧结果
                existing_ids = {c["id"] for c in chunks}
                for nc in new_chunks:
                    if nc["id"] not in existing_ids:
                        nc["score"] *= 0.7  # 重检索结果降权
                        chunks.append(nc)
                chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
                logger.info("Self-RAG 重检索: +%d 条新候选", len(new_chunks))

        return chunks

    # ============ 辅助方法 ============

    def _check_access(self, col_name: str, pt, user_can_see: List[str]) -> bool:
        if "internal" not in col_name:
            return True
        if not user_can_see:
            return False
        access = pt.payload.get("access_roles", "")
        if isinstance(access, str):
            access = [x.strip() for x in access.split(",")]
        return any(r in user_can_see for r in (access if isinstance(access, list) else [access]))

    def _load_chunks_by_ids(self, ids: List[str]) -> List[dict]:
        """批量加载 chunk 详情。"""
        chunks = []
        for col in [self._shared, self._internal]:
            try:
                # Qdrant 不支持一次 retrieve 超过 1000 个点
                for batch_start in range(0, len(ids), 500):
                    batch = ids[batch_start:batch_start + 500]
                    results = self.client.retrieve(
                        collection_name=col, ids=batch, with_payload=True
                    )
                    for pt in results:
                        chunks.append({
                            "id": pt.id,
                            "text": pt.payload.get("display_text", pt.payload.get("text", "")),
                            "score": 0.5,
                            "metadata": {k: v for k, v in pt.payload.items()
                                        if k not in ("text", "display_text")},
                            "doc_type": "internal" if "internal" in col else "shared",
                        })
            except Exception as e:
                logger.debug("加载 chunk 详情异常 [%s]: %s", col, e)
        return chunks

    def _expand_window(self, hits: List[dict], user_can_see: List[str],
                        max_extra: int = 50) -> List[dict]:
        """聚合查询窗口扩展。"""
        seen_ids = set(h["id"] for h in hits)
        doc_matches = {}
        for h in hits[:15]:
            fh = h["metadata"].get("file_hash", "")
            ci = h["metadata"].get("chunk_index", 0)
            if fh:
                doc_matches.setdefault(fh, []).append(ci)

        added = 0
        for fhash, indices in doc_matches.items():
            if added >= max_extra:
                break
            min_ci = max(0, min(indices) - 10)
            max_ci = max(indices) + 10
            for col_name in [self._shared, self._internal]:
                if added >= max_extra:
                    break
                try:
                    offset = None
                    while True:
                        page, offset = self.client.scroll(
                            collection_name=col_name, limit=1000,
                            with_payload=True, with_vectors=False, offset=offset,
                        )
                        for pt in page:
                            if pt.id in seen_ids:
                                continue
                            if pt.payload.get("file_hash") == fhash:
                                ci = pt.payload.get("chunk_index", -1)
                                if ci < min_ci or ci > max_ci:
                                    continue
                                if col_name == self._internal and user_can_see:
                                    access = pt.payload.get("access_roles", "")
                                    if isinstance(access, str):
                                        access = [x.strip() for x in access.split(",")]
                                    if not any(r in user_can_see for r in (access if isinstance(access, list) else [access])):
                                        continue
                                hits.append({
                                    "id": pt.id,
                                    "text": pt.payload.get("display_text", pt.payload.get("text", "")),
                                    "score": max(0.3, (hits[0]["score"] * 0.45) if hits else 0.3),
                                    "metadata": {k: v for k, v in pt.payload.items()
                                                if k not in ("text", "display_text")},
                                    "doc_type": "internal" if "internal" in col_name else "shared",
                                })
                                seen_ids.add(pt.id)
                                added += 1
                                if added >= max_extra:
                                    break
                        if offset is None:
                            break
                except Exception:
                    pass
        if added:
            logger.info("窗口扩展: +%d 条相邻片段", added)
            hits.sort(key=lambda h: h.get("score", 0), reverse=True)
        return hits

    # ============ 统计 & 查询 ============

    def file_hash_exists(self, file_hash: str, collection: str = None) -> bool:
        for col in ([collection] if collection else [self._shared, self._internal]):
            try:
                offset = None
                while True:
                    page, offset = self.client.scroll(
                        collection_name=col, limit=1000,
                        with_payload=["file_hash"], with_vectors=False, offset=offset)
                    for pt in page:
                        if pt.payload and pt.payload.get("file_hash") == file_hash:
                            return True
                    if offset is None:
                        break
            except Exception:
                pass
        return False

    def list_indexed_documents(self, collection: str = None) -> List[dict]:
        all_docs = []
        for col in ([collection] if collection else [self._shared, self._internal]):
            try:
                pts, offset = [], None
                while True:
                    page, offset = self.client.scroll(
                        collection_name=col, limit=1000,
                        with_payload=True, with_vectors=False, offset=offset)
                    pts.extend(page)
                    if offset is None:
                        break
                doc_map = {}
                for pt in pts:
                    pl = pt.payload or {}
                    fn = pl.get("file_name", "unknown")
                    if fn not in doc_map:
                        doc_map[fn] = {
                            "file_name": fn,
                            "total_pages": pl.get("total_pages", 0),
                            "file_hash": pl.get("file_hash", ""),
                            "file_type": pl.get("file_type", ""),
                            "chunk_count": 0,
                            "collection": col,
                            "access_roles": pl.get("access_roles", ""),
                            "report_date": pl.get("report_date", ""),
                        }
                    doc_map[fn]["chunk_count"] += 1
                all_docs.extend(doc_map.values())
            except Exception as e:
                logger.warning("列出文档异常 [%s]: %s", col, e)
        return sorted(all_docs, key=lambda d: d["file_name"])

    def get_file_name_by_hash(self, file_hash: str) -> Optional[str]:
        for col in [self._shared, self._internal]:
            try:
                offset = None
                while True:
                    page, offset = self.client.scroll(
                        collection_name=col, limit=1000,
                        with_payload=["file_hash", "file_name"],
                        with_vectors=False, offset=offset)
                    for pt in page:
                        if pt.payload and pt.payload.get("file_hash") == file_hash:
                            return pt.payload.get("file_name")
                    if offset is None:
                        break
            except Exception:
                pass
        return None

    def get_collection_stats(self) -> dict:
        try:
            si = self.client.get_collection(self._shared)
            ii = self.client.get_collection(self._internal)
            return {"shared_chunks": si.points_count, "internal_chunks": ii.points_count}
        except Exception:
            return {"shared_chunks": 0, "internal_chunks": 0}

    def _get_embed_dim(self) -> int:
        global _EMBED_DIM
        if _EMBED_DIM is None:
            emb = get_embedding_model()
            vec = _encode_query(emb, "dim")
            _EMBED_DIM = len(vec)
        return _EMBED_DIM

    # ============ RAPTOR 层次化摘要 ============

    async def build_raptor_summaries(self, chunks: List[str], file_name: str,
                                      file_hash: str, collection: str,
                                      metadata: dict):
        """RAPTOR: 对文档 chunks 做语义聚类 → LLM 生成摘要 → 摘要同原文一起索引。

        摘要作为特殊 chunk 存入 Qdrant，参与后续所有检索。
        """
        if len(chunks) < 5:
            return 0
        llm = get_llm()
        if llm is None:
            return 0

        try:
            from app.raptor_summarizer import RAPTORSummarizer
            raptor = RAPTORSummarizer(get_embedding_model())

            # 1. 聚类
            clusters = raptor.cluster_chunks(chunks, max_clusters=8)
            if len(clusters) <= 2:
                return 0

            # 2. 生成摘要
            summaries = await raptor.generate_summaries(
                chunks, clusters, file_name, llm
            )
            if not summaries:
                return 0

            # 3. 摘要入索引（tag 标记为 RAPTOR 摘要）
            summary_meta = dict(metadata)
            summary_meta["is_raptor_summary"] = True
            summary_meta["file_name"] = file_name
            summary_meta["file_hash"] = file_hash
            summary_meta["chunk_index"] = -1  # 摘要不是原始 chunk
            self.index_document(
                summaries, metadata=summary_meta,
                collection=collection, display_texts=summaries,
            )
            logger.info("RAPTOR: %s → %d 摘要已索引", file_name, len(summaries))
            return len(summaries)
        except Exception as e:
            logger.warning("RAPTOR 摘要构建失败: %s", e)
            return 0

    # ============ GraphRAG 构建（升级版：社区发现） ============

    async def build_knowledge_graph(self, file_hash: str, file_name: str, chunks: List[str]):
        """为文档构建知识图谱（异步，上传后调用），含社区检测。"""
        if not settings.graph_rag_enabled:
            return
        llm = get_llm()
        if llm is None:
            return
        try:
            from app.graph_rag import build_entity_graph
            count = await build_entity_graph(chunks, file_hash, file_name,
                                              self.entity_graph, llm)
            if count > 0:
                graph_path = settings.qdrant_path + "/entity_graph.json"
                import os
                os.makedirs(os.path.dirname(graph_path), exist_ok=True)
                self.entity_graph.save(graph_path)

                # 社区检测仅在 RAPTOR 模式开启时触发
                if settings.raptor_summaries and self.entity_graph.graph.number_of_nodes() % 50 < count:
                    await self._rebuild_communities(llm)

                logger.info("知识图谱已更新: %s → %d 实体", file_name, count)
        except Exception as e:
            logger.warning("知识图谱构建失败: %s", e)

    async def _rebuild_communities(self, llm):
        """运行社区检测 + 生成社区摘要。"""
        try:
            from app.raptor_summarizer import EnhancedGraphRAG
            enhanced = EnhancedGraphRAG(self.entity_graph)
            n_communities = enhanced.detect_communities()
            if n_communities > 0:
                await enhanced.generate_community_summaries(llm)
                # 保存增强版
                self._enhanced_graph = enhanced
                logger.info("GraphRAG 社区: %d 个检测完成", n_communities)
        except Exception as e:
            logger.debug("社区检测跳过: %s", e)

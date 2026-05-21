"""
混合检索 —— BM25(jieba) + BGE-M3 Sparse + Dense + ColBERT Late Interaction + RRF
"""
import math
import re
import logging
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import jieba

logger = logging.getLogger(__name__)

# ---- 制造业自定义词典 ----
_MFG_TERMS = [
    "基本工资", "岗位津贴", "加班费", "绩效奖金", "实发工资", "社保扣除", "公积金扣除",
    "个税", "工号", "全勤奖", "绩效工资", "社保扣款", "公积金扣款",
    "操作规程", "作业指导", "安全规范", "设备维护", "质量检验", "生产计划",
    "客户日报", "工作进度", "绩效考核", "薪酬制度", "工资表", "考勤",
    "SOP", "SMT", "CNC", "PLC", "QC", "QA", "BOM", "ERP", "MES",
]
for t in _MFG_TERMS:
    jieba.add_word(t)


def tokenize(text: str) -> List[str]:
    """jieba 分词，过滤标点和空白。"""
    tokens = jieba.lcut(text.lower())
    result = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if re.match(r'^[\s\-_\.\,\;\:\!\?\(\)\[\]\{\}\'\"\\\/\@\#\$\%\^\&\*\+\=]+$', t):
            continue
        if len(t) >= 1:
            result.append(t)
    return result


class InMemorySparseIndex:
    """BM25 内存索引（jieba 分词版）—— 标准 Okapi BM25 实现。"""

    def __init__(self):
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lengths: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._doc_texts: Dict[str, str] = {}
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0
        self._built: bool = False

    def index_point(self, point_id: str, text: str):
        tokens = tokenize(text)
        if not tokens:
            return
        doc_len = len(tokens)
        tf_map: Dict[str, int] = defaultdict(int)
        for t in tokens:
            tf_map[t] += 1
        for term, tf in tf_map.items():
            self.inverted_index[term].append((point_id, tf))
            self.doc_freqs[term] += 1
        self.doc_lengths[point_id] = doc_len
        self._doc_texts[point_id] = text
        self.total_docs += 1

    def remove_point(self, point_id: str):
        text = self._doc_texts.pop(point_id, None)
        if text is None:
            return
        tokens = tokenize(text)
        tf_map: Dict[str, int] = defaultdict(int)
        for t in tokens:
            tf_map[t] += 1
        for term in tf_map:
            if term in self.inverted_index:
                self.inverted_index[term] = [
                    (pid, tf) for pid, tf in self.inverted_index[term]
                    if pid != point_id
                ]
                if not self.inverted_index[term]:
                    del self.inverted_index[term]
                self.doc_freqs[term] = max(0, self.doc_freqs[term] - 1)
        self.doc_lengths.pop(point_id, None)
        self.total_docs = max(0, self.total_docs - 1)

    def build_from_texts(self, items: List[Tuple[str, str]]):
        """items: [(point_id, text), ...]"""
        self.reset()
        for pid, text in items:
            self.index_point(pid, text)
        if self.doc_lengths:
            self.avg_doc_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        self._built = True

    def search(self, query: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """标准 Okapi BM25 检索。"""
        if not self._built or self.total_docs == 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores: Dict[str, float] = defaultdict(float)
        k1, b = 1.5, 0.75
        for token in tokens:
            if token not in self.inverted_index:
                continue
            df = self.doc_freqs[token]
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            for point_id, tf in self.inverted_index[token]:
                doc_len = self.doc_lengths.get(point_id, self.avg_doc_len)
                if self.avg_doc_len > 0:
                    norm = doc_len / self.avg_doc_len
                else:
                    norm = 1.0
                bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * norm))
                scores[point_id] += idf * bm25_tf
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def reset(self):
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        self.inverted_index.clear()
        self._doc_texts.clear()
        self.total_docs = 0
        self.avg_doc_len = 0.0
        self._built = False


def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: Optional[List[Tuple[str, float]]] = None,
    keyword_results: Optional[List[Tuple[str, float]]] = None,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 0.8,
    keyword_weight: float = 0.5,
    top_k: int = 50,
) -> List[Tuple[str, float]]:
    """三路 RRF 融合：Dense + BGE-M3 Sparse + 经典 BM25 关键词。"""
    scores: Dict[str, float] = defaultdict(float)
    for rank, (pid, _) in enumerate(dense_results, start=1):
        scores[pid] += dense_weight / (k + rank)
    if sparse_results:
        for rank, (pid, _) in enumerate(sparse_results, start=1):
            scores[pid] += sparse_weight / (k + rank)
    if keyword_results:
        for rank, (pid, _) in enumerate(keyword_results, start=1):
            scores[pid] += keyword_weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class BGE_M3_SparseRetriever:
    """BGE-M3 原生稀疏向量检索 —— 使用 Qdrant 稀疏向量搜索。"""

    def __init__(self, qdrant_client, embed_model):
        self._client = qdrant_client
        self._embed = embed_model

    def encode_sparse(self, text: str) -> Tuple[List[int], List[float]]:
        """用 BGE-M3 生成稀疏向量 (lexical weights)。"""
        output = self._embed.encode([text], return_sparse=True)
        sparse_vec = output['lexical_weights'][0]
        indices = list(sparse_vec.keys())
        values = [float(v) for v in sparse_vec.values()]
        return indices, values

    def search(self, query: str, collection: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """在 Qdrant 中执行稀疏向量搜索。"""
        from qdrant_client.http import models as qm
        indices, values = self.encode_sparse(query)
        if not indices:
            return []
        try:
            results = self._client.search(
                collection_name=collection,
                query_vector=qm.NamedSparseVector(
                    name="sparse",
                    vector=qm.SparseVector(indices=indices, values=values),
                ),
                limit=top_k,
                with_payload=False,
            )
            return [(r.id, r.score) for r in results]
        except Exception as e:
            logger.warning("稀疏向量搜索失败: %s", e)
            return []


class ColBERTLateInteraction:
    """ColBERT 风格的 MaxSim 迟交互重排序。

    原理：对 query 和每个候选 chunk 的 token-level embeddings 计算 MaxSim 分数，
    实现细粒度的 token 级语义匹配，比 single-vector 余弦相似度精准得多。
    """

    def __init__(self, embed_model):
        self._embed = embed_model

    def encode_tokens(self, texts: List[str]) -> List[np.ndarray]:
        """获取文本的 ColBERT token embeddings。

        返回: List of [num_tokens, dim] numpy arrays
        """
        output = self._embed.encode(texts, return_colbert_vecs=True)
        return output['colbert_vecs']

    def score(self, query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
        """MaxSim 分数：对 query 的每个 token，找 doc 中最相似的 token，求和。

        query_vecs: [Q_dim] or [Q_tokens, dim]
        doc_vecs: [D_tokens, dim]

        返回: float, MaxSim 分数（越高越相关）
        """
        if query_vecs.ndim == 1:
            query_vecs = query_vecs.reshape(1, -1)
        if doc_vecs.ndim == 1:
            doc_vecs = doc_vecs.reshape(1, -1)

        # 归一化
        q_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
        d_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)

        # [Q, D] 相似度矩阵
        sim_matrix = np.dot(q_norm, d_norm.T)  # [Q_tokens, D_tokens]

        # MaxSim: sum over query tokens of max similarity to any doc token
        max_per_query = sim_matrix.max(axis=1)  # [Q_tokens]
        return float(max_per_query.sum())

    def rerank(self, query: str, candidates: List[Tuple[str, str, float]],
               top_k: int = 20) -> List[Tuple[str, float]]:
        """用 ColBERT MaxSim 重排候选列表。

        candidates: [(id, text, initial_score), ...]
        返回: [(id, final_score), ...]
        """
        if not candidates:
            return []

        cand_texts = [c[1] for c in candidates]
        try:
            q_vecs = self.encode_tokens([query])[0]
            d_vecs_list = self.encode_tokens(cand_texts)

            scored = []
            for i, (pid, _text, init_score) in enumerate(candidates):
                ms = self.score(q_vecs, d_vecs_list[i])
                # 融合 MaxSim 和初始分数
                combined = 0.7 * ms + 0.3 * (init_score if init_score else 0.5)
                scored.append((pid, combined))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.warning("ColBERT 重排失败: %s，降级为原始排序", e)
            return [(c[0], c[2]) for c in candidates[:top_k]]

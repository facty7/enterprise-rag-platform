"""
RAPTOR 层次化摘要 —— 来自 RAGFlow 的核心算法

原理：
1. 将 chunks 按语义相似度聚类（GMM 自适应聚类）
2. 每类用 LLM 生成一个摘要 chunk
3. 摘要与原始 chunks 一起向量化索引
4. 查询时，摘要提供"鸟瞰视角"，原始 chunks 提供细节

这解决了 Enterprise RAG 当前最大的弱点：汇总/全局类问题查询时需要在 20+ 个 chunks 中扫描，
而 RAPTOR 的摘要可以直接回答这类问题。
"""
import logging
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """你是一位企业知识管理专家。以下是同一主题下的多个文档片段，请将它们汇总为一个简洁的摘要（150-300字）。
摘要应包含关键数据、结论和实体名称，格式如下：

[摘要] 文档名: {doc_name}
关键信息：...

文档片段：
{chunks}

摘要："""


class RAPTORSummarizer:
    """RAPTOR 风格层次化摘要器。

    对每个文档的 chunks 做语义聚类 → LLM 生成摘要 → 摘要同原文一起索引。
    """

    def __init__(self, embed_model):
        self._embed = embed_model

    def cluster_chunks(self, chunks: List[str], max_clusters: int = 8) -> List[List[int]]:
        """用 GMM 自适应聚类 chunks。

        返回: [[chunk_idx, ...], ...] 每个子列表是一个类的成员索引
        """
        n = len(chunks)
        if n <= 3:
            # 太少不分簇
            return [list(range(n))]

        # 向量化
        try:
            output = self._embed.encode(chunks, return_dense=True)
            if isinstance(output, dict):
                vectors = output['dense_vecs']
            else:
                vectors = np.array(output)
        except Exception:
            vectors = self._embed.encode(chunks)
            if isinstance(vectors, dict):
                vectors = vectors['dense_vecs']
            vectors = np.array(vectors)

        # 自定聚类数
        n_clusters = min(max_clusters, max(2, n // 3))
        if n_clusters < 2:
            return [list(range(n))]

        try:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=3)
            labels = gmm.fit_predict(vectors)
        except Exception:
            # GMM 失败 → 简单按长度分组
            indices = list(range(n))
            indices.sort(key=lambda i: len(chunks[i]))
            k = min(n_clusters, n)
            per = max(1, n // k)
            return [indices[i:i + per] for i in range(0, n, per)]

        # 按 label 分组
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(i)

        result = list(clusters.values())
        logger.info("RAPTOR 聚类: %d chunks → %d 簇", n, len(result))
        return result

    async def generate_summaries(self, chunks: List[str], clusters: List[List[int]],
                                  doc_name: str, llm, max_chunks_per_summary: int = 8) -> List[str]:
        """为每个簇生成摘要。"""
        if llm is None:
            return []

        summaries = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue  # 单 chunk 不摘要

            # 取簇中代表性的 chunks（最长 + 中等长度各取几个）
            selected = sorted(cluster, key=lambda i: len(chunks[i]), reverse=True)
            selected = selected[:max_chunks_per_summary]
            cluster_texts = [f"[片段{i}] {chunks[i][:500]}" for i in selected]

            prompt = _SUMMARIZE_PROMPT.format(
                doc_name=doc_name,
                chunks="\n\n".join(cluster_texts),
            )

            try:
                resp = await llm.acomplete(prompt)
                summary = str(resp).strip()
                if len(summary) >= 30:
                    summary = f"[文档: {doc_name}][RAPTOR摘要] {summary}"
                    summaries.append(summary)
            except Exception as e:
                logger.debug("RAPTOR摘要生成失败: %s", e)

        if summaries:
            logger.info("RAPTOR: %s → %d 摘要", doc_name, len(summaries))
        return summaries


class EnhancedGraphRAG:
    """社区发现 GraphRAG —— 来自 RAGFlow 的进阶图谱。

    在原有实体-文档图基础上，增加：
    1. Louvain 社区检测 —— 发现实体群落
    2. 社区摘要生成 —— LLM 总结每个社区的主题
    3. 跨文档推理 —— 同一社区的实体即使在不同文档中也能关联
    """

    def __init__(self, entity_graph):
        self.graph = entity_graph
        self.communities: Dict[int, List[str]] = {}  # community_id → [entity_names]
        self.community_summaries: Dict[int, str] = {}  # community_id → summary
        self._built = False

    def detect_communities(self) -> int:
        """用 Louvain 算法检测实体社区。"""
        import networkx as nx
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(self.graph.graph, seed=42)
        except ImportError:
            # Fallback: 连通分量
            communities = list(nx.connected_components(self.graph.graph))

        self.communities = {}
        for i, comm in enumerate(communities):
            self.communities[i] = list(comm)

        self._built = True
        logger.info("GraphRAG 社区检测: %d 个社区", len(self.communities))
        return len(self.communities)

    async def generate_community_summaries(self, llm) -> int:
        """为每个社区生成摘要。"""
        if llm is None or not self._built:
            return 0

        count = 0
        for cid, entities in self.communities.items():
            if len(entities) < 2:
                continue

            # 取代表性实体（按连接数排序）
            degrees = [(e, self.graph.graph.degree(e)) for e in entities if e in self.graph.graph]
            degrees.sort(key=lambda x: x[1], reverse=True)
            top_entities = [e for e, _ in degrees[:10]]

            # 获取这些实体出现的文档
            doc_names = set()
            for e in top_entities:
                for fh in self.graph.entity_docs.get(e, set()):
                    doc_names.add(fh[:8] if len(fh) > 8 else fh)  # 用 hash 前8位

            prompt = f"""以下是知识图谱中同一社区的关键实体和它们之间的关系：

实体: {', '.join(top_entities)}
相关文档: {len(doc_names)} 个

请用一句话总结这个社区的主题（这个社区在讨论什么？30字以内）：

主题："""

            try:
                resp = await llm.acomplete(prompt)
                summary = str(resp).strip()
                if len(summary) >= 4:
                    self.community_summaries[cid] = summary
                    count += 1
            except Exception as e:
                logger.debug("社区摘要生成失败: %s", e)

        if count:
            logger.info("GraphRAG 社区摘要: %d 个", count)
        return count

    def get_community_context(self, entity_name: str) -> str:
        """获取某实体所属社区的信息。"""
        for cid, entities in self.communities.items():
            if entity_name in entities:
                summary = self.community_summaries.get(cid, "")
                if summary:
                    return f"[图谱社区] {summary} (含 {len(entities)} 个关联实体)"
        return ""

    def build_context_for_query(self, query: str, max_communities: int = 3) -> str:
        """为查询构建社区级知识图谱上下文。"""
        if not self._built:
            return ""

        # 匹配查询中的实体
        matched_entities = self.graph.search_entities(query, top_k=10)

        # 找到这些实体所属的社区
        seen_communities = set()
        parts = []
        for entity in matched_entities[:5]:
            for cid, entities in self.communities.items():
                if entity in entities and cid not in seen_communities:
                    seen_communities.add(cid)
                    summary = self.community_summaries.get(cid, "")
                    if summary and len(parts) < max_communities:
                        related = [e for e in entities if e != entity][:5]
                        parts.append(
                            f"● 社区: {summary}\n"
                            f"  关联实体: {', '.join(related)}"
                        )
                    break

        return "\n".join(parts) if parts else ""

"""
GraphRAG —— 轻量级知识图谱增强检索

实现思路：
1. 文档上传时，LLM 从 chunk 中提取实体（人名、公司、项目、产品、日期等）
2. 构建实体-文档-片段的三层图
3. 查询时：
   a. 从 query 中提取实体
   b. 查找图中关联的实体、文档、片段
   c. 将关联实体信息注入 Prompt，辅助跨文档推理

这是 Microsoft GraphRAG 的简化版，复杂度可控但效果显著。
"""
import json
import logging
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

import networkx as nx

logger = logging.getLogger(__name__)

_ENTITY_EXTRACT_PROMPT = """从以下文本片段中提取关键实体（人名、公司名、项目名、产品名、职位、地点、日期、金额等）。
输出 JSON 格式，只输出 JSON，不要其他内容。

文本：{text}

JSON 格式：
{{"entities": [{{"name": "实体名", "type": "person|company|project|product|position|location|date|amount|other"}}, ...]}}
如果没有实体，输出 {{"entities": []}}"""


class EntityGraph:
    """轻量级实体-文档知识图谱。"""

    def __init__(self):
        self.graph = nx.Graph()
        self.entity_types: Dict[str, str] = {}         # entity_name → type
        self.entity_docs: Dict[str, Set[str]] = defaultdict(set)  # entity_name → {file_hash, ...}
        self.doc_entities: Dict[str, Set[str]] = defaultdict(set)  # file_hash → {entity_name, ...}

    def add_entities(self, file_hash: str, file_name: str, entities: List[dict]):
        """将提取的实体加入图谱。"""
        for ent in entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "other")
            if not name or len(name) < 2:
                continue
            if name not in self.graph:
                self.graph.add_node(name, type=etype)
            self.entity_types[name] = etype
            self.entity_docs[name].add(file_hash)
            self.doc_entities[file_hash].add(name)
            # 连接实体到文档
            if file_name not in self.graph:
                self.graph.add_node(file_name, type="document")
            self.graph.add_edge(name, file_name, relation="mentioned_in")

        # 同一文档的实体之间建立共现关系
        doc_ents = [e.get("name", "").strip() for e in entities if e.get("name", "").strip()]
        for i, e1 in enumerate(doc_ents):
            for e2 in doc_ents[i + 1:]:
                if e1 != e2:
                    if self.graph.has_edge(e1, e2):
                        self.graph[e1][e2]["weight"] = self.graph[e1][e2].get("weight", 1) + 0.5
                    else:
                        self.graph.add_edge(e1, e2, weight=1.0, relation="co_occurrence")

    def get_related(self, entity_name: str, max_hops: int = 2) -> List[dict]:
        """获取与某实体相关的实体和文档。"""
        if entity_name not in self.graph:
            return []
        related = []
        for node in nx.single_source_shortest_path_length(self.graph, entity_name, cutoff=max_hops):
            if node == entity_name:
                continue
            etype = self.entity_types.get(node, self.graph.nodes[node].get("type", ""))
            related.append({"name": node, "type": etype})
        return related

    def search_entities(self, query_text: str, top_k: int = 10) -> List[str]:
        """从查询文本中匹配已知实体。"""
        matched = []
        query_lower = query_text.lower()
        for entity in self.graph.nodes:
            if len(entity) >= 2 and entity.lower() in query_lower:
                etype = self.entity_types.get(entity, "unknown")
                docs = self.entity_docs.get(entity, set())
                matched.append((entity, etype, len(docs)))
        matched.sort(key=lambda x: x[2], reverse=True)
        return [m[0] for m in matched[:top_k]]

    def build_context_for_query(self, query: str, max_entities: int = 5,
                                 max_related: int = 15) -> str:
        """为查询构建知识图谱上下文。"""
        matched = self.search_entities(query, top_k=max_entities)
        if not matched:
            return ""

        parts = []
        seen = set()
        for entity in matched[:max_entities]:
            etype = self.entity_types.get(entity, "未知")
            parts.append(f"● {entity}（{etype}）")
            seen.add(entity)

            related = self.get_related(entity, max_hops=1)
            count = 0
            for r in related:
                if r["name"] not in seen and count < max_related:
                    parts.append(f"  ↳ 关联: {r['name']}（{r['type']}）")
                    seen.add(r["name"])
                    count += 1

        return "\n".join(parts) if parts else ""

    def stats(self) -> dict:
        return {
            "total_entities": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": {t: sum(1 for n in self.graph.nodes if self.entity_types.get(n) == t)
                            for t in set(self.entity_types.values())},
        }

    def save(self, path: str):
        """保存图谱到文件。"""
        data = {
            "entities": {n: {"type": self.entity_types.get(n, "unknown")}
                        for n in self.graph.nodes},
            "edges": [{"source": u, "target": v, "weight": d.get("weight", 1)}
                     for u, v, d in self.graph.edges(data=True)],
            "entity_docs": {k: list(v) for k, v in self.entity_docs.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        """从文件加载图谱。"""
        import os
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph.clear()
        for n, info in data["entities"].items():
            self.graph.add_node(n, type=info.get("type", "unknown"))
            self.entity_types[n] = info.get("type", "unknown")
        for e in data["edges"]:
            self.graph.add_edge(e["source"], e["target"], weight=e.get("weight", 1))
        for k, v in data["entity_docs"].items():
            self.entity_docs[k] = set(v)
            for fh in v:
                self.doc_entities[fh].add(k)
        logger.info("GraphRAG 图谱加载: %d 实体, %d 边", self.graph.number_of_nodes(), self.graph.number_of_edges())


async def extract_entities_from_chunk(chunk_text: str, llm) -> List[dict]:
    """从文本片段中提取实体。"""
    try:
        prompt = _ENTITY_EXTRACT_PROMPT.format(text=chunk_text[:800])
        resp = await llm.acomplete(prompt)
        text = str(resp).strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            data = json.loads(m.group())
            return data.get("entities", [])
    except Exception as e:
        logger.debug("实体提取失败: %s", e)
    return []


async def build_entity_graph(chunks: List[str], file_hash: str, file_name: str,
                              graph: EntityGraph, llm, batch_size: int = 10) -> int:
    """批量提取实体并构建知识图谱。返回新增实体数。"""
    total_entities = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # 采样处理（每批取代表性片段）
        samples = batch[:5] if len(batch) > 5 else batch
        for chunk in samples:
            entities = await extract_entities_from_chunk(chunk, llm)
            if entities:
                graph.add_entities(file_hash, file_name, entities)
                total_entities += len(entities)
    if total_entities > 0:
        logger.info("GraphRAG: 从 %s 提取 %d 个实体", file_name, total_entities)
    return total_entities

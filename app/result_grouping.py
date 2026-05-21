"""
结果分组 —— 按文档聚合片段 + 简单多样性排序。
"""
from typing import List, Dict


def group_by_document(hits: List[dict]) -> List[dict]:
    """将检索结果按文档分组，组内按 chunk_index 排序。"""
    groups: Dict[str, dict] = {}
    for h in hits:
        meta = h.get("metadata", {})
        fhash = meta.get("file_hash", "_unknown")
        if fhash not in groups:
            groups[fhash] = {
                "file_name": meta.get("file_name", ""),
                "file_hash": fhash,
                "chunks": [],
                "best_score": 0.0,
            }
        g = groups[fhash]
        g["chunks"].append(h)
        if h.get("score", 0) > g["best_score"]:
            g["best_score"] = h["score"]
    # 组内按 chunk_index 排序
    for g in groups.values():
        g["chunks"].sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
    # 组间按 best_score 排序
    result = sorted(groups.values(), key=lambda g: g["best_score"], reverse=True)
    return result


def apply_diversity(hits: List[dict], lambda_d: float = 0.7, top_k: int = 10) -> List[dict]:
    """MMR 多样性：避免同一文档独占 Top-K。"""
    if len(hits) <= top_k:
        return hits
    selected, remaining = [], list(hits)
    # 取第一个
    selected.append(remaining.pop(0))
    while len(selected) < top_k and remaining:
        best_idx, best_score = 0, -1e9
        for i, h in enumerate(remaining):
            rel = h.get("score", 0)
            # 同文档惩罚
            same_doc_penalty = 0.0
            for s in selected:
                if s.get("metadata", {}).get("file_hash") == h.get("metadata", {}).get("file_hash"):
                    same_doc_penalty = 1.0
                    break
            mmr = lambda_d * rel - (1 - lambda_d) * same_doc_penalty
            if mmr > best_score:
                best_score, best_idx = mmr, i
        selected.append(remaining.pop(best_idx))
    return selected

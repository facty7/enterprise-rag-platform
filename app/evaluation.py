"""
RAG 评估框架 —— RAGAS + Hit Rate + MRR + NDCG + 自定义指标

提供：
1. RAGAS 自动化评估（faithfulness, answer_relevancy, context_precision, context_recall）
2. 检索质量评估（Hit Rate, MRR, NDCG）
3. 人工标注测试集管理
4. 自动回归测试
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """检索质量指标计算。"""

    @staticmethod
    def hit_rate(retrieved_ids: List[str], relevant_ids: List[str], k: int = None) -> float:
        """Hit Rate @ K：前 K 个结果中命中相关文档的比例。"""
        if k:
            retrieved_ids = retrieved_ids[:k]
        hits = len(set(retrieved_ids) & set(relevant_ids))
        return hits / max(len(relevant_ids), 1) if relevant_ids else 0.0

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """MRR (Mean Reciprocal Rank)：第一个相关结果的倒数排名。"""
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int = 10) -> float:
        """NDCG @ K：归一化折损累计增益。"""
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k], 1):
            rel = relevance_scores.get(rid, 0.0)
            dcg += rel / np.log2(i + 1)

        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_scores, 1):
            idcg += rel / np.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def recall(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Recall：召回了多少相关文档。"""
        hits = len(set(retrieved_ids) & set(relevant_ids))
        return hits / max(len(relevant_ids), 1) if relevant_ids else 0.0

    @staticmethod
    def precision(retrieved_ids: List[str], relevant_ids: List[str], k: int = None) -> float:
        """Precision @ K：前 K 个结果中相关的比例。"""
        if k:
            retrieved_ids = retrieved_ids[:k]
        hits = len(set(retrieved_ids) & set(relevant_ids))
        return hits / max(len(retrieved_ids), 1) if retrieved_ids else 0.0


class EvalTestSuite:
    """测试集管理。

    测试集格式:
    {
        "name": "测试集名称",
        "created": "2026-05-07",
        "tests": [
            {
                "query": "用户问题",
                "relevant_docs": ["file_hash_1", "file_hash_2"],
                "relevant_chunks": ["chunk_id_1"],
                "expected_answer_contains": ["关键词1", "关键词2"],
                "relevance_scores": {"chunk_id_1": 0.9, "chunk_id_2": 0.7},
                "difficulty": "easy|medium|hard",
                "category": "事实查询|汇总统计|概念解释|操作流程"
            }
        ]
    }
    """

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tests: List[dict] = []

    def add_test(self, query: str, relevant_docs: List[str] = None,
                 relevant_chunks: List[str] = None,
                 expected_answer_contains: List[str] = None,
                 relevance_scores: Dict[str, float] = None,
                 difficulty: str = "medium",
                 category: str = "事实查询"):
        self.tests.append({
            "query": query,
            "relevant_docs": relevant_docs or [],
            "relevant_chunks": relevant_chunks or [],
            "expected_answer_contains": expected_answer_contains or [],
            "relevance_scores": relevance_scores or {},
            "difficulty": difficulty,
            "category": category,
        })

    def save(self, name: str):
        data = {
            "name": name,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "count": len(self.tests),
            "tests": self.tests,
        }
        path = self.save_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("测试集已保存: %s (%d 条)", path, len(self.tests))

    def load(self, name: str):
        path = self.save_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"测试集不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tests = data["tests"]
        logger.info("测试集已加载: %s (%d 条)", name, len(self.tests))


class RAGEvaluator:
    """RAG 系统端到端评估器。"""

    def __init__(self, rag_engine, llm, embed_model):
        self.rag = rag_engine
        self.llm = llm
        self.embed = embed_model

    async def evaluate_retrieval(self, test_suite: EvalTestSuite) -> Dict[str, float]:
        """评估检索质量（Hit Rate, MRR, NDCG, Recall, Precision）。"""
        metrics = RetrievalMetrics()
        all_hr, all_mrr, all_ndcg, all_recall, all_prec = [], [], [], [], []

        for test in test_suite.tests:
            query = test["query"]
            relevant_ids = test.get("relevant_chunks", [])
            relevance_scores = test.get("relevance_scores", {})

            # 执行检索
            results = await self.rag.retrieve(query)
            retrieved_ids = [r["id"] for r in results]

            all_hr.append(metrics.hit_rate(retrieved_ids, relevant_ids, k=10))
            all_mrr.append(metrics.mrr(retrieved_ids, relevant_ids))
            all_ndcg.append(metrics.ndcg(retrieved_ids, relevance_scores, k=10))
            all_recall.append(metrics.recall(retrieved_ids, relevant_ids))
            all_prec.append(metrics.precision(retrieved_ids, relevant_ids, k=10))

        return {
            "hit_rate@10": np.mean(all_hr),
            "mrr": np.mean(all_mrr),
            "ndcg@10": np.mean(all_ndcg),
            "recall": np.mean(all_recall),
            "precision@10": np.mean(all_prec),
            "num_tests": len(test_suite.tests),
        }

    async def evaluate_generation(self, test_suite: EvalTestSuite) -> Dict[str, float]:
        """评估生成质量（答案包含预期关键词的命中率）。"""
        keyword_hits = []
        for test in test_suite.tests:
            query = test["query"]
            expected_keywords = test.get("expected_answer_contains", [])
            if not expected_keywords:
                continue

            try:
                # 检索
                results = await self.rag.retrieve(query)
                chunks = [{"id": r["id"], "text": r["text"], "score": r["score"],
                           "metadata": r.get("metadata", {})} for r in results]

                # 构建 prompt 并生成答案
                from app.prompt_builder import build_qa_prompt
                from app.result_grouping import group_by_document, apply_diversity

                hits = apply_diversity(chunks, top_k=10)
                doc_groups = group_by_document(hits)
                prompt = build_qa_prompt(query, doc_groups, max_chunks=15)
                resp = await self.llm.acomplete(prompt)
                answer = str(resp)

                # 检查关键词命中
                hits = sum(1 for kw in expected_keywords if kw in answer)
                keyword_hits.append(hits / max(len(expected_keywords), 1))
            except Exception as e:
                logger.warning("评估生成失败: %s", e)
                keyword_hits.append(0.0)

        return {
            "keyword_recall": np.mean(keyword_hits) if keyword_hits else 0.0,
            "num_evaluated": len(keyword_hits),
        }

    async def evaluate_ragas(self, test_suite: EvalTestSuite) -> dict:
        """使用 RAGAS 框架评估（如果已安装）。

        指标: faithfulness, answer_relevancy, context_precision, context_recall
        """
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from datasets import Dataset

            eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
            for test in test_suite.tests[:20]:  # 限制数量避免过长
                query = test["query"]
                results = await self.rag.retrieve(query)
                contexts = [r["text"][:500] for r in results[:5]]

                from app.prompt_builder import build_qa_prompt
                from app.result_grouping import group_by_document, apply_diversity

                hits = apply_diversity(results, top_k=10)
                doc_groups = group_by_document(hits)
                prompt = build_qa_prompt(query, doc_groups)
                resp = await self.llm.acomplete(prompt)
                answer = str(resp)

                eval_data["question"].append(query)
                eval_data["answer"].append(answer)
                eval_data["contexts"].append(contexts)
                eval_data["ground_truth"].append(test.get("expected_answer_contains", ""))

            dataset = Dataset.from_dict(eval_data)
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                                context_precision, context_recall])
            return {k: float(v) for k, v in result.items()}
        except ImportError:
            logger.warning("RAGAS 未安装，跳过 RAGAS 评估")
            return {"error": "ragas not installed"}
        except Exception as e:
            logger.warning("RAGAS 评估失败: %s", e)
            return {"error": str(e)}

    async def full_evaluation(self, test_suite: EvalTestSuite) -> dict:
        """执行完整评估：检索 + 生成 + RAGAS。"""
        logger.info("========== 开始全量评估 (%d 条测试) ==========", len(test_suite.tests))
        t0 = time.time()

        retrieval_metrics = await self.evaluate_retrieval(test_suite)
        generation_metrics = await self.evaluate_generation(test_suite)
        # ragas_metrics = await self.evaluate_ragas(test_suite)  # 可选，较慢

        elapsed = time.time() - t0
        logger.info("评估完成，耗时 %.1f 秒", elapsed)

        return {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            # "ragas": ragas_metrics,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }


class ComparisonReporter:
    """与旧系统对比报告生成器。"""

    @staticmethod
    def compare(old_metrics: dict, new_metrics: dict) -> str:
        """生成对比报告。"""
        lines = [
            "=" * 60,
            "  Baseline RAG 旧系统 vs Enterprise RAG 新系统 —— 检索质量对比",
            "=" * 60,
            "",
            f"{'指标':<25} {'旧系统':>10} {'新系统':>10} {'变化':>10}",
            "-" * 60,
        ]

        # 检索指标
        for key in ["hit_rate@10", "mrr", "ndcg@10", "recall", "precision@10"]:
            old_val = old_metrics.get("retrieval", {}).get(key, 0)
            new_val = new_metrics.get("retrieval", {}).get(key, 0)
            if old_val > 0:
                change = f"+{(new_val - old_val) / old_val * 100:.0f}%"
            else:
                change = "N/A"
            lines.append(f"{key:<25} {old_val:>10.4f} {new_val:>10.4f} {change:>10}")

        # 生成指标
        old_kr = old_metrics.get("generation", {}).get("keyword_recall", 0)
        new_kr = new_metrics.get("generation", {}).get("keyword_recall", 0)
        if old_kr > 0:
            change = f"+{(new_kr - old_kr) / old_kr * 100:.0f}%"
        else:
            change = "N/A"
        lines.append(f"{'keyword_recall':<25} {old_kr:>10.4f} {new_kr:>10.4f} {change:>10}")

        lines.append("-" * 60)
        lines.append(f"旧系统评估耗时: {old_metrics.get('elapsed_seconds', 0):.1f}s")
        lines.append(f"新系统评估耗时: {new_metrics.get('elapsed_seconds', 0):.1f}s")
        lines.append("=" * 60)

        return "\n".join(lines)

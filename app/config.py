"""Enterprise RAG platform configuration."""
import os
import hashlib
import base64
from pathlib import Path

from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_obfuscation_key() -> bytes:
    seed = "EnterpriseRAG_2026_SecureLayer_v2_FixedKey"
    return hashlib.sha256(seed.encode()).digest()


def decrypt_api_key(encrypted: str) -> str:
    if not encrypted:
        return ""
    try:
        raw = base64.b64decode(encrypted)
        salt = raw[:12]
        data = raw[12:]
        key = _get_obfuscation_key()
        ks = b""
        c = 0
        while len(ks) < len(data):
            ks += hashlib.sha256(key + salt + c.to_bytes(4, "big")).digest()
            c += 1
        return bytes(a ^ b for a, b in zip(data, ks[:len(data)])).decode("utf-8")
    except Exception:
        return ""


def encrypt_api_key(plaintext: str) -> str:
    import secrets
    salt = secrets.token_bytes(12)
    key = _get_obfuscation_key()
    data = plaintext.encode("utf-8")
    ks = b""
    c = 0
    while len(ks) < len(data):
        ks += hashlib.sha256(key + salt + c.to_bytes(4, "big")).digest()
        c += 1
    encrypted = bytes(a ^ b for a, b in zip(data, ks[:len(data)]))
    return base64.b64encode(salt + encrypted).decode()


class Settings(BaseSettings):
    # Qdrant
    qdrant_mode: str = "local"
    qdrant_path: str = str(PROJECT_ROOT / "data" / "qdrant_db")
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""

    # Embedding —— BGE-M3 (1024d dense + sparse lexical + ColBERT multi-vector)
    # 离线部署可在 .env 中改为 models/BAAI--bge-m3。
    embed_model_name: str = "BAAI/bge-m3"
    embed_device: str = "cpu"
    embed_use_fp16: bool = False

    # Reranker —— bge-reranker-v2-m3（离线部署可改为本地模型路径）
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = True

    # LLM
    llm_api_base: str = "https://api.deepseek.com"
    llm_model_name: str = "deepseek-v4-flash"
    llm_api_key: str = ""
    llm_api_key_enc: str = ""

    # 高级检索特性开关
    hybrid_search_enabled: bool = True
    hyde_enabled: bool = True           # 假设文档嵌入
    multi_query_enabled: bool = True    # 多查询融合
    multi_query_count: int = 3          # 生成几个变体查询
    late_interaction_enabled: bool = True  # ColBERT MaxSim (BGE-M3)
    self_rag_enabled: bool = True       # Self-RAG 质量评估循环
    graph_rag_enabled: bool = True      # 知识图谱增强
    semantic_chunking: bool = True      # 语义分块 (BGE-M3 dense)
    raptor_summaries: bool = False      # RAPTOR 层次摘要 (文档多时开启，小规模反而干扰)
    query_decompose_enabled: bool = True  # 复杂查询分解
    search_mode_default: str = "auto"  # 默认检索模式：fast / balanced / deep / debug / auto
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 600
    metrics_enabled: bool = True
    hf_hub_offline: bool = False
    transformers_offline: bool = False

    # 文档处理
    chunk_size: int = 800               # 语义分块最大长度
    chunk_overlap: int = 100            # 语义分块重叠
    top_k_retrieval: int = 20           # 最终返回数量
    retrieval_candidates: int = 50      # 粗排候选数
    context_prefix_enabled: bool = True

    # RRF 融合
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 0.8
    keyword_weight: float = 0.5

    # Late Interaction (ColBERT)
    colbert_top_k: int = 30             # ColBERT 重排候选数

    # Graph RAG
    graph_max_entities: int = 500
    graph_community_threshold: float = 0.6

    # Self-RAG
    self_rag_threshold: float = 0.5     # 相关性阈值，低于此值触发重检索
    self_rag_max_rounds: int = 2        # 最大重检索轮次

    # 限额（大幅提升）
    max_documents: int = 2000
    max_chunks: int = 50000
    max_entities: int = 5000

    # 文件监视
    watch_folder: str = str(PROJECT_ROOT / "files")
    upload_path: str = str(PROJECT_ROOT / "data" / "uploads")
    reports_path: str = str(PROJECT_ROOT / "data" / "reports")

    # 用户数据库
    user_db_key: str = ""

    # 服务
    host: str = "0.0.0.0"
    port: int = 8400
    log_level: str = "INFO"

    # 评估
    eval_test_dir: str = str(PROJECT_ROOT / "data" / "eval_tests")

    def get_llm_api_key(self) -> str:
        if self.llm_api_key_enc:
            return decrypt_api_key(self.llm_api_key_enc)
        return self.llm_api_key

    @property
    def shared_collection(self) -> str:
        return "kb_enterprise_shared"

    @property
    def internal_collection(self) -> str:
        return "kb_enterprise_internal"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False, "extra": "ignore"}


settings = Settings()

os.environ.setdefault("HF_HUB_OFFLINE", "1" if settings.hf_hub_offline else "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1" if settings.transformers_offline else "0")


SEARCH_MODE_PRESETS = {
    "auto": {
        "label": "自动",
        "description": "根据问题类型自动选择轻链路或深链路",
        "query_rewrite": True,
        "hyde": False,
        "multi_query": False,
        "query_decompose": False,
        "late_interaction": True,
        "self_rag": False,
        "graph_rag": False,
        "query_variants_limit": 2,
        "retrieval_candidates": 40,
        "context_chunks": 20,
        "dynamic": True,
    },
    "fast": {
        "label": "快速",
        "description": "低延迟轻链路，适合简单事实问题",
        "query_rewrite": False,
        "hyde": False,
        "multi_query": False,
        "query_decompose": False,
        "late_interaction": False,
        "self_rag": False,
        "graph_rag": False,
        "query_variants_limit": 1,
        "retrieval_candidates": 30,
        "context_chunks": 12,
    },
    "balanced": {
        "label": "平衡",
        "description": "默认交付模式，兼顾延迟与召回",
        "query_rewrite": True,
        "hyde": False,
        "multi_query": False,
        "query_decompose": False,
        "late_interaction": True,
        "self_rag": False,
        "graph_rag": False,
        "query_variants_limit": 2,
        "retrieval_candidates": 40,
        "context_chunks": 20,
    },
    "deep": {
        "label": "深度",
        "description": "复杂查询增强链路，启用重策略",
        "query_rewrite": True,
        "hyde": True,
        "multi_query": True,
        "query_decompose": True,
        "late_interaction": True,
        "self_rag": True,
        "graph_rag": True,
        "query_variants_limit": 4,
        "retrieval_candidates": 50,
        "context_chunks": 30,
    },
    "debug": {
        "label": "调试",
        "description": "深度模式 + 详细链路追踪",
        "query_rewrite": True,
        "hyde": True,
        "multi_query": True,
        "query_decompose": True,
        "late_interaction": True,
        "self_rag": True,
        "graph_rag": True,
        "query_variants_limit": 4,
        "retrieval_candidates": 50,
        "context_chunks": 30,
    },
}

SEARCH_MODE_ALIASES = {
    "automatic": "auto",
    "default": "balanced",
    "standard": "balanced",
}


def normalize_search_mode(mode: str | None) -> str:
    raw = (mode or settings.search_mode_default or "balanced").strip().lower()
    raw = SEARCH_MODE_ALIASES.get(raw, raw)
    if raw not in SEARCH_MODE_PRESETS:
        return settings.search_mode_default if settings.search_mode_default in SEARCH_MODE_PRESETS else "balanced"
    return raw


def choose_search_mode(query: str, requested_mode: str | None = None) -> str:
    normalized = normalize_search_mode(requested_mode)
    if normalized != "auto":
        return normalized

    q = (query or "").strip()
    if not q:
        return "balanced"

    long_like = len(q) > 40
    aggregate_keywords = [
        "汇总", "统计", "对比", "比较", "分别", "全部", "所有", "哪些", "多少", "排名",
        "流程", "步骤", "怎么", "如何", "原因", "为什么", "差异", "趋势", "明细",
    ]
    if long_like or any(k in q for k in aggregate_keywords):
        return "deep"
    if any(k in q for k in ["什么是", "解释", "定义", "查询", "找", "在哪", "多少", "谁", "哪"]):
        return "balanced"
    return "fast"


def get_search_mode_profile(mode: str | None) -> dict:
    normalized = normalize_search_mode(mode)
    profile = dict(SEARCH_MODE_PRESETS[normalized])
    profile["name"] = normalized
    return profile

"""Enterprise RAG knowledge base service."""
import logging
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response

from app.config import settings
from app.metrics import CONTENT_TYPE_LATEST, generate_latest, update_indexed_docs
from app.routes import router
from app.auth import validate_token

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("fitz").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("========== Enterprise RAG 知识库 v3.0 ==========")

    # 预热：加载嵌入模型并初始化向量库
    try:
        from app.rag_engine import get_embedding_model, get_rag_engine, _has_flag_embedding
        emb = get_embedding_model()
        try:
            output = emb.encode(["probe"], return_dense=True)
            dim = output['dense_vecs'].shape[1]
            logger.info("BGE-M3 就绪: %dd (dense+sparse+colbert)", dim)
        except Exception:
            dim = len(emb.get_text_embedding("probe"))
            logger.info("嵌入模型就绪: %dd (dense only)", dim)

        rag = get_rag_engine()
        stats = rag.get_collection_stats()
        update_indexed_docs(stats)
        logger.info("Qdrant 就绪: 共享 %d 条, 内部 %d 条",
                     stats["shared_chunks"], stats["internal_chunks"])
    except Exception as exc:
        logger.error("模型预热失败: %s", exc)

    # LLM 连通性检查
    try:
        from app.rag_engine import get_llm
        llm = get_llm()
        if llm:
            logger.info("DeepSeek API 连接正常 (模型: %s)", settings.llm_model_name)
        else:
            logger.warning("LLM 未配置")
    except Exception:
        logger.warning("LLM 连接检查跳过")

    # 文件监视器
    watcher = None
    if settings.watch_folder:
        try:
            from app.folder_watcher import FolderWatcher, make_watcher_callback
            cb = make_watcher_callback(get_rag_engine())
            watcher = FolderWatcher(settings.watch_folder, cb)
            watcher.start()
        except Exception as exc:
            logger.warning("监视器启动失败: %s", exc)

    # 打印检索特性
    logger.info("========== 检索特性 ==========")
    logger.info("  BGE-M3 (dense+sparse+colbert): %s", "Y" if _has_flag_embedding() else "N")
    logger.info("  HyDE: %s", "Y" if settings.hyde_enabled else "N")
    logger.info("  Multi-Query: %s", "Y" if settings.multi_query_enabled else "N")
    logger.info("  ColBERT Late Interaction: %s", "Y" if settings.late_interaction_enabled else "N")
    logger.info("  Self-RAG: %s", "Y" if settings.self_rag_enabled else "N")
    logger.info("  GraphRAG: %s", "Y" if settings.graph_rag_enabled else "N")
    logger.info("  Semantic Chunking: %s", "Y" if settings.semantic_chunking else "N")
    logger.info("  Cross-encoder Reranker: %s", "Y" if settings.reranker_enabled else "N")
    logger.info("  Search mode default: %s", settings.search_mode_default)
    logger.info("================================")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); lan_ip = s.getsockname()[0]; s.close()
    except Exception:
        lan_ip = "127.0.0.1"
    logger.info("  手机访问: http://%s:%d", lan_ip, settings.port)
    logger.info("==========================================")

    yield
    if watcher:
        watcher.stop()
    try:
        get_rag_engine().close()
    except Exception as exc:
        logger.debug("RAG engine close skipped: %s", exc)
    logger.info("========== Enterprise RAG 已关闭 ==========")


app = FastAPI(title="Enterprise RAG 知识库", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("异常: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "服务内部错误"})


app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "Enterprise RAG 3.0"}


@app.get("/metrics")
async def metrics():
    if not settings.metrics_enabled:
        return Response("metrics disabled\n", media_type="text/plain")
    try:
        from app.rag_engine import get_rag_engine
        update_indexed_docs(get_rag_engine().get_collection_stats())
    except Exception:
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if STATIC_DIR.exists():
    if (STATIC_DIR / "login.html").exists():
        @app.get("/login.html")
        async def serve_login():
            return FileResponse(STATIC_DIR / "login.html")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path:
            file_path = STATIC_DIR / full_path
            if file_path.is_file():
                return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")

    logger.info("前端已挂载 -> http://localhost:%d", settings.port)

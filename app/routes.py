"""Enterprise RAG API routes."""
import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, UploadFile, Request, Form
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel, Field

from app.config import settings
from app.config import get_search_mode_profile, normalize_search_mode, choose_search_mode
from app.cache import get_json_cache, make_cache_key, set_json_cache
from app.document_parser import parse_document
from app.metrics import CONTENT_TYPE_LATEST, generate_latest, observe_cache, observe_chat, observe_trace_phases
from app.rag_engine import get_rag_engine, get_llm, get_embedding_model, _has_flag_embedding
from app.text_splitter import split_text_into_chunks, split_text_into_chunks_with_pages
from app.chunking import enrich_chunks, semantic_split
from app.qa_preprocess import generate_qa_pairs
from app.prompt_builder import build_qa_prompt, build_fallback_context
from app.result_grouping import group_by_document, apply_diversity
from app.query_processor import expand_query, extract_keywords, is_aggregate_query
from app.reranker import rerank
from app.auth import (verify_password, create_session, validate_token, check_token,
                      can_access, can_upload_to, add_user, edit_user, delete_user,
                      change_password, change_name,
                      update_role_quotas, get_role_max, audit_log, read_audit_log,
                      get_role_counts, get_role_label, list_all_users, get_all_users_full,
                      ROLE_INFO)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
_rag = get_rag_engine()

OVERWRITE_PATTERNS = ["客户信息表", "每日报表"]


# ---------- 模型 ----------
class LoginRequest(BaseModel):
    username: str
    password: str


class UploadResponse(BaseModel):
    status: str = "ok"
    file_name: str
    file_hash: str = ""
    pages_parsed: int = 0
    chunks_indexed: int = 0
    duplicate: bool = False
    overwritten: bool = False
    collection: str = "shared"
    message: str


class DocumentInfo(BaseModel):
    file_name: str; file_hash: str = ""; total_pages: int
    chunk_count: int; file_type: str = ""
    collection: str = "shared"; access_roles: str = ""
    report_date: str = ""


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="auto", max_length=16)


class AddUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)
    role: str


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _clean_hit(hit: dict) -> dict:
    meta = hit.get("metadata", {}) or {}
    return {
        "id": str(hit.get("id", "")),
        "text": str(hit.get("text", "")),
        "score": float(hit.get("score", 0) or 0),
        "metadata": dict(meta),
        "doc_type": hit.get("doc_type", "shared"),
        "graph_context": hit.get("graph_context", ""),
    }


def _is_overwrite_file(filename: str) -> bool:
    return any(p in filename for p in OVERWRITE_PATTERNS)


# ---------- 登录 ----------
@router.post("/login")
async def login(body: LoginRequest):
    user = verify_password(body.username, body.password)
    if not user: raise HTTPException(status_code=401, detail="账号或密码错误")
    token = create_session(user)
    return {"token": token, "name": user["name"], "role": user["role"],
            "can_see": user["can_see"], "username": user["username"]}


@router.get("/me")
async def get_me(request: Request):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    return {"name": session["name"], "role": session["role"], "can_see": session["can_see"]}


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6)


class ChangeNameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)


@router.put("/me/password")
async def update_my_password(request: Request, body: ChangePasswordRequest):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    try:
        change_password(session["username"], body.old_password, body.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit_log(session["name"], session["role"], "change_password", "", _client_ip(request))
    return {"status": "ok", "message": "密码已修改，请重新登录"}


@router.put("/me/name")
async def update_my_name(request: Request, body: ChangeNameRequest):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    try:
        updated = change_name(session["username"], body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit_log(session["name"], session["role"], "change_name", body.name, _client_ip(request))
    return updated


# ---------- 上传 ----------
@router.post("/document/upload")
async def upload_document(request: Request, file: UploadFile = File(...),
                           collection: str = Form("shared"), access_roles: str = Form(""),
                           report_date: str = Form("")):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)

    if not file.filename: raise HTTPException(status_code=400, detail="文件名为空")
    file.filename = os.path.basename(file.filename)

    if collection == "internal" and not can_upload_to(session, "internal"):
        raise HTTPException(status_code=403, detail="无权限上传内部文档")
    if access_roles and session["role"] != "management" and access_roles not in session.get("can_see", []):
        raise HTTPException(status_code=403, detail=f"无权上传标记为「{access_roles}」的文档")

    target_col = settings.internal_collection if collection == "internal" else settings.shared_collection

    try: file_bytes = await file.read()
    except Exception: raise HTTPException(status_code=400, detail="文件读取失败")
    if not file_bytes: raise HTTPException(status_code=400, detail="文件为空")

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    ext = os.path.splitext(file.filename)[1].lower()

    # 覆盖模式
    overwritten = False
    if collection == "internal" and _is_overwrite_file(file.filename):
        old_count = _rag.delete_by_filename(file.filename, target_col)
        if old_count > 0:
            overwritten = True
            for rd in (settings.upload_path, settings.watch_folder):
                root = Path(rd)
                if not root.exists(): continue
                for f in root.iterdir():
                    if f.is_dir():
                        for sf in f.iterdir():
                            if sf.is_file() and sf.name == file.filename:
                                try: shutil.rmtree(f, ignore_errors=True)
                                except Exception: pass
                                break
                    elif f.is_file() and f.name == file.filename:
                        try: f.unlink()
                        except Exception: pass

    # 保存文件
    saved_dir = os.path.join(settings.upload_path, file_hash)
    if not os.path.exists(os.path.join(saved_dir, file.filename)):
        os.makedirs(saved_dir, exist_ok=True)
        with open(os.path.join(saved_dir, file.filename), "wb") as f: f.write(file_bytes)

    if not overwritten and _rag.file_hash_exists(file_hash, target_col):
        return UploadResponse(file_name=file.filename, file_hash=file_hash,
                              duplicate=True, collection=collection,
                              message=f"「{file.filename}」已存在")

    try: doc = parse_document(file_bytes, file.filename)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc))
    if not doc.full_text.strip(): raise HTTPException(status_code=422, detail="无有效文本")

    if not report_date:
        m = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', file.filename)
        if m: report_date = m.group(1).replace('_', '-')

    base_meta = {"file_name": file.filename, "total_pages": doc.total_pages,
                 "file_hash": file_hash, "file_type": ext,
                 "access_roles": access_roles, "report_date": report_date,
                 "uploaded_by": session["name"], "collection": collection}

    # 语义分块
    if ext == '.pdf' and doc.total_pages > 0:
        cwp = split_text_into_chunks_with_pages(doc.full_text)
        if not cwp: raise HTTPException(status_code=422, detail="切分无结果")
        chunks = [c[0] for c in cwp]; pcm = [{"page": c[1]} for c in cwp]
    else:
        if settings.semantic_chunking and _has_flag_embedding():
            try:
                chunks = semantic_split(doc.full_text, get_embedding_model())
            except Exception:
                chunks = split_text_into_chunks(doc.full_text)
        else:
            chunks = split_text_into_chunks(doc.full_text)
        if not chunks: raise HTTPException(status_code=422, detail="切分无结果")
        pcm = None

    sheet = list(doc.table_headers.keys())[0] if doc.table_headers else ""
    header = doc.table_headers.get(sheet, "") if sheet else ""
    display_chunks = enrich_chunks(chunks, file.filename, ext,
                                  sheet_name=sheet, table_header=header,
                                  per_chunk_meta=pcm)

    # QA 预处理
    qa_chunks = None
    try:
        import asyncio
        chunks_with_qa = await generate_qa_pairs(chunks, file.filename)
        if len(chunks_with_qa) > len(chunks):
            qa_display = list(display_chunks)
            for i in range(len(chunks), len(chunks_with_qa)):
                qa_display.append(f"[文档: {file.filename}] {chunks_with_qa[i]}")
            chunks = chunks_with_qa
            display_chunks = qa_display
    except Exception:
        pass

    try:
        count = _rag.index_document(
            chunks, metadata=base_meta, per_chunk_meta=pcm,
            collection=target_col, display_texts=display_chunks,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 异步构建知识图谱 + RAPTOR（文档数 > 50 时自动开启层次化摘要）
    try:
        import asyncio
        asyncio.create_task(_rag.build_knowledge_graph(file_hash, file.filename, chunks))

        # 自动检测：总文档数超过 50 时启用 RAPTOR
        total_docs = len(_rag.list_indexed_documents())
        should_raptor = settings.raptor_summaries or total_docs > 50
        if should_raptor and len(chunks) >= 5:
            asyncio.create_task(_rag.build_raptor_summaries(
                chunks, file.filename, file_hash, target_col, base_meta
            ))
    except Exception:
        pass

    audit_log(session["name"], session["role"], "upload", file.filename, _client_ip(request))
    return UploadResponse(file_name=file.filename, file_hash=file_hash,
                          pages_parsed=doc.total_pages, chunks_indexed=count,
                          overwritten=overwritten, collection=collection,
                          message=f"「{file.filename}」{'已覆盖更新' if overwritten else '处理完成'}（{count} 个片段）")


# ---------- 文档列表 ----------
@router.get("/documents")
async def list_documents(request: Request):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    all_docs = _rag.list_indexed_documents()
    result = []
    for d in all_docs:
        if d["collection"] == settings.shared_collection:
            result.append(d)
        elif can_access(session, d.get("access_roles", "")):
            result.append(d)
    return [DocumentInfo(**d) for d in result]


@router.delete("/document/{file_hash}")
async def delete_document(request: Request, file_hash: str):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] not in ("management", "manager"):
        raise HTTPException(status_code=403, detail="仅管理层和业务经理可删除")
    file_name = _rag.get_file_name_by_hash(file_hash)
    ok = _rag.delete_document(file_hash)
    if not ok: raise HTTPException(status_code=404, detail="文档不存在")
    for rd in (settings.upload_path, settings.watch_folder):
        root = Path(rd)
        if not root.exists(): continue
        hd = root / file_hash
        if hd.is_dir(): shutil.rmtree(hd, ignore_errors=True)
        if file_name and (root / file_name).is_file():
            try: (root / file_name).unlink()
            except Exception: pass
    audit_log(session["name"], session["role"], "delete", file_name or file_hash, _client_ip(request))
    return {"status": "ok", "message": "已删除"}


@router.get("/reports")
async def list_reports(request: Request, date: str = ""):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if not date: date = datetime.now().strftime("%Y-%m-%d")
    all_docs = _rag.list_indexed_documents(settings.internal_collection)
    return [DocumentInfo(**d) for d in all_docs
            if d.get("report_date") == date and can_access(session, "reports")]


# ---------- 文件下载 ----------
_MIME = {".pdf":"application/pdf",".xlsx":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         ".xls":"application/vnd.ms-excel",".docx":"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
         ".doc":"application/msword",".pptx":"application/vnd.openxmlformats-officedocument.presentationml.presentation",
         ".ppt":"application/vnd.ms-powerpoint",".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",
         ".txt":"text/plain; charset=utf-8",".csv":"text/csv; charset=utf-8"}


@router.get("/file/{file_hash}")
async def serve_file(request: Request, file_hash: str):
    all_docs = _rag.list_indexed_documents()
    target_doc = None
    for d in all_docs:
        if d.get("file_hash") == file_hash:
            target_doc = d; break
    if target_doc and target_doc["collection"] != settings.shared_collection:
        session = check_token(request)
        if not session: raise HTTPException(status_code=401, detail="请先登录后再下载内部文档")
        if not can_access(session, target_doc.get("access_roles", "")):
            raise HTTPException(status_code=403, detail="无权访问此文件")
    for rd in (settings.upload_path, settings.watch_folder):
        root = Path(rd)
        if not root.exists(): continue
        hd = root / file_hash
        if hd.is_dir():
            for f in hd.iterdir():
                if f.is_file():
                    return FileResponse(f, filename=f.name, media_type=_MIME.get(f.suffix.lower(), "application/octet-stream"))
        for f in root.iterdir():
            if f.is_file() and f.name.startswith(file_hash):
                return FileResponse(f, filename=f.name, media_type=_MIME.get(f.suffix.lower(), "application/octet-stream"))
    raise HTTPException(status_code=404)


@router.get("/status")
async def system_status(request: Request):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    stats = _rag.get_collection_stats()
    graph_stats = {}
    try:
        graph_stats = _rag.entity_graph.stats() if _rag._entity_graph else {}
    except Exception:
        pass
    return {"shared_chunks": stats["shared_chunks"], "internal_chunks": stats["internal_chunks"],
            "llm_configured": get_llm() is not None,
            "version": "Enterprise RAG 3.0", "retrieval_features": {
                "bge_m3": _has_flag_embedding(),
                "hyde": settings.hyde_enabled,
                "multi_query": settings.multi_query_enabled,
                "late_interaction": settings.late_interaction_enabled,
                "self_rag": settings.self_rag_enabled,
                "graph_rag": settings.graph_rag_enabled,
                "semantic_chunking": settings.semantic_chunking,
            },
            "graph_stats": graph_stats,
            "user": session["name"], "role": session["role"]}


# ---------- QR ----------
QR_LOCK_FILE = Path(__file__).resolve().parent.parent / "config" / "qr_lock.txt"


@router.get("/qr")
async def qr_code():
    import socket
    from app.qr_generator import generate_qr_png
    from fastapi.responses import Response

    def _detect_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    current_ip = _detect_ip()
    locked_ip = ""
    if QR_LOCK_FILE.exists():
        locked_ip = QR_LOCK_FILE.read_text().strip()

    def _same_subnet(a, b):
        try:
            pa, pb = a.split("."), b.split(".")
            return len(pa) == 4 and len(pb) == 4 and pa[:3] == pb[:3]
        except Exception:
            return False

    ip = locked_ip if (locked_ip and _same_subnet(locked_ip, current_ip)) else current_ip
    if not locked_ip or not _same_subnet(locked_ip, current_ip):
        QR_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        QR_LOCK_FILE.write_text(ip)

    return Response(content=generate_qr_png(f"http://{ip}:{settings.port}"), media_type="image/png")


@router.get("/qr-url")
async def qr_url():
    import socket
    ip = ""
    if QR_LOCK_FILE.exists():
        ip = QR_LOCK_FILE.read_text().strip()
    if not ip:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        except Exception:
            ip = "127.0.0.1"
    return {"url": f"http://{ip}:{settings.port}"}


# ---------- 管理员 ----------
@router.get("/admin/users")
async def admin_list_users(request: Request):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] != "management": raise HTTPException(status_code=403, detail="仅管理层可访问")
    users = list_all_users()
    role_counts = get_role_counts()
    role_info = {}
    for r, info in ROLE_INFO.items():
        role_info[r] = {"label": info["label"], "max": get_role_max(r)}
    return {"users": users, "role_counts": role_counts, "role_info": role_info}


@router.put("/admin/roles")
async def admin_update_roles(request: Request, body: dict):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] != "management": raise HTTPException(status_code=403, detail="仅管理层可操作")
    try:
        result = update_role_quotas(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit_log(session["name"], session["role"], "update_quotas", str(body), _client_ip(request))
    return result


@router.post("/admin/users")
async def admin_add_user(request: Request, body: AddUserRequest):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] != "management": raise HTTPException(status_code=403, detail="仅管理层可添加用户")
    try:
        new_user = add_user(body.name, body.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit_log(session["name"], session["role"], "add_user",
              f"{new_user['username']} ({new_user['name']})", _client_ip(request))
    return {"username": new_user["username"], "name": new_user["name"],
            "role": new_user["role"], "password": new_user["password"]}


class EditUserRequest(BaseModel):
    name: str = None; password: str = None; role: str = None
    can_see: list = None; new_username: str = None


@router.put("/admin/users/{username}")
async def admin_edit_user(request: Request, username: str, body: EditUserRequest):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] != "management": raise HTTPException(status_code=403, detail="仅管理层可操作")
    try:
        updated = edit_user(username, name=body.name, password=body.password,
                           role=body.role, can_see=body.can_see, new_username=body.new_username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit_log(session["name"], session["role"], "edit_user",
              f"{username} (fields: {[k for k,v in body.dict().items() if v is not None]})",
              _client_ip(request))
    return {"username": updated["username"], "name": updated["name"],
            "role": updated["role"], "can_see": updated["can_see"]}


@router.delete("/admin/users/{username}")
async def admin_delete_user(request: Request, username: str):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] != "management": raise HTTPException(status_code=403, detail="仅管理层可操作")
    if username == session["username"]: raise HTTPException(status_code=400, detail="不能删除自己")
    try:
        deleted = delete_user(username)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit_log(session["name"], session["role"], "delete_user",
              f"{username} ({deleted['name']})", _client_ip(request))
    return {"status": "ok", "message": f"已删除用户「{deleted['name']}」", "user": deleted}


class AdminViewUsersRequest(BaseModel):
    password: str


@router.post("/admin/users-full")
async def admin_view_all_users(request: Request, body: AdminViewUsersRequest):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] != "management": raise HTTPException(status_code=403, detail="仅管理层可访问")
    admin_user = verify_password(session["username"], body.password)
    if not admin_user: raise HTTPException(status_code=401, detail="管理员密码错误")
    audit_log(session["name"], session["role"], "view_all_users", "", _client_ip(request))
    return {"users": get_all_users_full()}


@router.get("/admin/audit")
async def admin_audit_log(request: Request, limit: int = 100):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] != "management": raise HTTPException(status_code=403, detail="仅管理层可访问")
    return read_audit_log(limit)


# ---------- 流式问答 ----------
@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    request_started = time.perf_counter()
    cache_event = "disabled"

    can_see = list(session.get("can_see", ["shared"]))
    if "shared" not in can_see:
        can_see.append("shared")
    if session.get("role") not in can_see:
        can_see.append(session["role"])

    search_query = body.query
    effective_top_k = body.top_k or settings.top_k_retrieval
    requested_mode = normalize_search_mode(body.mode)
    requested_profile = get_search_mode_profile(requested_mode)
    strategy_mode = choose_search_mode(body.query, body.mode)
    mode_profile = get_search_mode_profile(strategy_mode)
    if is_aggregate_query(body.query):
        effective_top_k = max(effective_top_k, 25)

    trace = {
        "mode": strategy_mode,
        "requested_mode": requested_mode,
        "requested_mode_label": requested_profile["label"],
        "requested_mode_description": requested_profile["description"],
        "mode_label": mode_profile["label"],
        "mode_description": mode_profile["description"],
        "candidate_count": mode_profile["retrieval_candidates"],
        "query_variants_count": 0,
        "final_count": 0,
        "total_ms": 0.0,
        "timing_ms": {},
        "phase": {},
        "flags": {},
        "query_variants": [],
        "cache": "miss",
    }

    # 异步检索
    cache_key = make_cache_key("retrieve", {
        "query": search_query,
        "top_k": effective_top_k * 2,
        "can_see": sorted(can_see),
        "mode": requested_mode,
        "strategy_mode": strategy_mode,
    })
    cached = get_json_cache(cache_key) if settings.redis_enabled else None
    if body.query.strip():
        if cached:
            hits = cached.get("hits", [])
            trace = cached.get("trace", trace)
            trace["cache"] = "hit"
            cache_event = "hit"
        else:
            hits, trace = await _rag.retrieve(
                query=search_query,
                top_k=effective_top_k * 2,
                user_can_see=can_see,
                strategy_mode=strategy_mode,
                return_trace=True,
            )
            trace["cache"] = "miss"
            if settings.redis_enabled:
                cache_event = "miss"
                set_json_cache(cache_key, {
                    "hits": [_clean_hit(h) for h in hits],
                    "trace": trace,
                })
        trace["requested_mode"] = requested_mode
        trace["requested_mode_label"] = requested_profile["label"]
        trace["requested_mode_description"] = requested_profile["description"]
        trace["actual_mode"] = strategy_mode
        trace["actual_mode_label"] = mode_profile["label"]
        trace["actual_mode_description"] = mode_profile["description"]
    else:
        hits = []

    # 多样性重排
    if hits:
        hits = apply_diversity(hits, top_k=effective_top_k)
    doc_groups = group_by_document(hits) if hits else []
    llm = get_llm()

    # 获取 GraphRAG 上下文（优先用检索结果中附带的上下文）
    graph_context = ""
    if hits and hits[0].get("graph_context"):
        graph_context = hits[0]["graph_context"]

    async def event_generator() -> AsyncGenerator[str, None]:
        status = "ok"
        seen_hashes, sources = set(), []
        try:
            for g in doc_groups:
                fh = g.get("file_hash", "")
                if fh and fh not in seen_hashes:
                    seen_hashes.add(fh)
                    best_chunk = None
                    for c in g.get("chunks", []):
                        if best_chunk is None or c.get("score", 0) > best_chunk.get("score", 0):
                            best_chunk = c
                    best_meta = best_chunk.get("metadata", {}) if best_chunk else {}
                    page = best_meta.get("page", 0)
                    chunk_index = best_meta.get("chunk_index", 0)
                    snippet = (best_chunk.get("text", "") if best_chunk else "").strip().replace("\n", " ")
                    if len(snippet) > 180:
                        snippet = snippet[:180].rstrip() + "…"
                    sources.append({"file_name": g.get("file_name", ""),
                                    "file_hash": fh,
                                    "score": round(g.get("best_score", 0), 4),
                                    "collection": g.get("chunks", [{}])[0].get("doc_type", "shared") if g.get("chunks") else "shared",
                                    "page": page,
                                    "chunk_index": chunk_index,
                                    "snippet": snippet})
            sources.sort(key=lambda s: s["score"], reverse=True)
            sources = sources[:5]
            yield f"data: [SOURCES]{json.dumps(sources, ensure_ascii=False)}[/SOURCES]\n\n"
            trace_payload = dict(trace)
            trace_payload["query_variants"] = trace_payload.get("query_variants", [])[:4]
            yield f"data: [TRACE]{json.dumps(trace_payload, ensure_ascii=False)}[/TRACE]\n\n"

            if llm is None:
                fallback = build_fallback_context(hits)
                yield f"data: {fallback}\n\n"
                yield "data: [DONE]\n\n"; return

            max_c = mode_profile["context_chunks"]
            if is_aggregate_query(body.query) and strategy_mode != "fast":
                max_c = min(max_c + 10, 40)
            # 意图检测
            from app.query_processor import detect_intent
            intent = detect_intent(body.query)
            prompt = build_qa_prompt(body.query, doc_groups, max_chunks=max_c,
                                     graph_context=graph_context, intent=intent)
            stream = await llm.astream_complete(prompt)
            async for chunk in stream:
                delta = chunk.delta if hasattr(chunk, "delta") else str(chunk)
                if delta: yield f"data: {delta}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            status = "error"
            yield f"data: [ERROR] AI 服务异常: {exc}\n\n"
        finally:
            observe_cache(cache_event)
            observe_trace_phases(trace)
            observe_chat(requested_mode, strategy_mode, status, time.perf_counter() - request_started)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ---------- 对话历史 ----------
HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "chat_history.json"
MAX_CONVERSATIONS = 10


def _load_history() -> dict:
    if not HISTORY_FILE.exists(): return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.loads(f.read() or "{}")
    except Exception:
        return {}


def _save_history(data: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


class SaveConversationRequest(BaseModel):
    id: str; title: str = ""; msgs: list = []


@router.get("/conversations")
async def list_conversations(request: Request):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    convs = _load_history().get(session["username"], [])
    return {"conversations": [{"id": c["id"], "title": c["title"], "ts": c["ts"],
                               "n": len(c["msgs"])} for c in convs]}


@router.get("/conversations/{conv_id}")
async def get_conversation(request: Request, conv_id: str):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    convs = _load_history().get(session["username"], [])
    for c in convs:
        if c["id"] == conv_id: return c
    raise HTTPException(status_code=404, detail="对话不存在")


@router.post("/conversations")
async def save_conversation(request: Request, body: SaveConversationRequest):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if not body.id or not body.msgs: raise HTTPException(status_code=400, detail="参数错误")
    all_h = _load_history()
    convs = all_h.get(session["username"], [])
    now = datetime.now().strftime("%m-%d %H:%M")
    for c in convs:
        if c["id"] == body.id:
            c["msgs"] = body.msgs; c["title"] = body.title or c["title"]
            c["ts"] = now; _save_history(all_h); return {"status": "ok"}
    convs.insert(0, {"id": body.id, "title": body.title, "ts": now, "msgs": body.msgs})
    if len(convs) > MAX_CONVERSATIONS: convs = convs[:MAX_CONVERSATIONS]
    all_h[session["username"]] = convs
    _save_history(all_h)
    return {"status": "ok"}


@router.delete("/conversations/{conv_id}")
async def delete_conversation(request: Request, conv_id: str):
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    all_h = _load_history()
    convs = all_h.get(session["username"], [])
    all_h[session["username"]] = [c for c in convs if c["id"] != conv_id]
    _save_history(all_h)
    return {"status": "ok"}


# ---------- 评估接口 ----------
@router.get("/eval/status")
async def eval_status(request: Request):
    """查看评估功能状态。"""
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    features = {
        "eval_available": True,
        "ragas_available": False,
        "metrics_supported": ["hit_rate", "mrr", "ndcg", "recall", "precision", "keyword_recall"],
    }
    try:
        import ragas; features["ragas_available"] = True
    except ImportError:
        pass
    return features


@router.post("/eval/run")
async def run_evaluation(request: Request, body: dict):
    """运行评估。body: {"test_name": "test_suite_name"}"""
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    if session["role"] != "management":
        raise HTTPException(status_code=403, detail="仅管理层可运行评估")

    test_name = body.get("test_name", "default")
    from app.evaluation import EvalTestSuite, RAGEvaluator
    suite = EvalTestSuite(settings.eval_test_dir)
    try:
        suite.load(test_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"测试集不存在: {test_name}")

    evaluator = RAGEvaluator(_rag, get_llm(), get_embedding_model())
    result = await evaluator.full_evaluation(suite)
    return result


@router.get("/eval/tests")
async def list_eval_tests(request: Request):
    """列出所有评估测试集。"""
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    test_dir = Path(settings.eval_test_dir)
    if not test_dir.exists():
        return {"tests": []}
    tests = []
    for f in test_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            tests.append({"name": data.get("name", f.stem), "count": data.get("count", 0),
                         "created": data.get("created", "")})
        except Exception:
            pass
    return {"tests": tests}


# ---------- GraphRAG 接口 ----------
@router.get("/graph/stats")
async def graph_stats(request: Request):
    """查看知识图谱统计信息。"""
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    try:
        return _rag.entity_graph.stats()
    except Exception:
        return {"error": "知识图谱未构建"}


@router.get("/graph/entity/{name}")
async def graph_entity(request: Request, name: str):
    """查询特定实体的关联信息。"""
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    try:
        related = _rag.entity_graph.get_related(name)
        return {"entity": name, "type": _rag.entity_graph.entity_types.get(name, "unknown"),
                "related": related}
    except Exception:
        raise HTTPException(status_code=404, detail="实体不存在或图谱未构建")


# ---------- 检索特性开关 ----------
@router.get("/retrieval/features")
async def retrieval_features(request: Request):
    """查看检索特性开关状态。"""
    session = validate_token(request)
    if not session: raise HTTPException(status_code=401)
    from app.config import SEARCH_MODE_PRESETS
    return {
        "bge_m3": _has_flag_embedding(),
        "hyde": settings.hyde_enabled,
        "multi_query": settings.multi_query_enabled,
        "late_interaction": settings.late_interaction_enabled,
        "self_rag": settings.self_rag_enabled,
        "graph_rag": settings.graph_rag_enabled,
        "semantic_chunking": settings.semantic_chunking,
        "reranker": settings.reranker_enabled,
        "reranker_model": settings.reranker_model_name,
        "embed_model": settings.embed_model_name,
        "llm_model": settings.llm_model_name,
        "search_mode_default": settings.search_mode_default,
        "search_modes": SEARCH_MODE_PRESETS,
    }

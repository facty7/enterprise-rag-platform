"""
文件夹监视器 —— 文件拖入即自动入库
"""
import hashlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)
_DEBOUNCE_SECONDS = 2.0


class FileHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self._callback = callback
        self._pending: dict[str, float] = {}
        self._timer: Optional[threading.Timer] = None

    def on_created(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def _enqueue(self, path: str):
        self._pending[path] = time.time()
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(_DEBOUNCE_SECONDS + 0.5, self._flush)
        self._timer.start()

    def _flush(self):
        now = time.time()
        for path, ts in list(self._pending.items()):
            if now - ts >= _DEBOUNCE_SECONDS:
                del self._pending[path]
                logger.info("检测到文件变更: %s", path)
                try:
                    self._callback(path)
                except Exception as exc:
                    logger.error("处理文件出错: %s: %s", path, exc)


class FolderWatcher:
    def __init__(self, watch_dir: str, callback: Callable[[str], None]):
        self._watch_dir = watch_dir
        self._callback = callback
        self._observer: Optional[Observer] = None

    def start(self):
        if not os.path.isdir(self._watch_dir):
            os.makedirs(self._watch_dir, exist_ok=True)
            logger.info("已创建监视目录: %s", self._watch_dir)
        self._observer = Observer()
        self._observer.schedule(FileHandler(self._callback), self._watch_dir, recursive=True)
        self._observer.start()
        logger.info("文件夹监视器已启动: %s", self._watch_dir)
        self._scan_existing()

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            logger.info("文件夹监视器已停止")

    def _scan_existing(self):
        root = Path(self._watch_dir)
        for f in root.glob("**/*"):
            if f.is_file():
                try:
                    self._callback(str(f))
                except Exception:
                    pass


def make_watcher_callback(rag_engine):
    from app.document_parser import parse_document
    from app.text_splitter import split_text_into_chunks, split_text_into_chunks_with_pages
    from app.chunking import enrich_chunks

    def handle_file(file_path: str):
        file_name = os.path.basename(file_path)
        stem = os.path.splitext(file_name)[0]
        if re.fullmatch(r"[0-9a-f]{64}", stem):
            return
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        if not file_bytes:
            return
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if rag_engine.file_hash_exists(file_hash):
            return
        # 拷贝到 uploads 目录，确保后续可下载
        from app.config import settings
        saved_dir = os.path.join(settings.upload_path, file_hash)
        os.makedirs(saved_dir, exist_ok=True)
        dest = os.path.join(saved_dir, file_name)
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(file_bytes)
        try:
            doc = parse_document(file_bytes, file_name)
        except ValueError:
            return
        if not doc.full_text.strip():
            return

        ext = os.path.splitext(file_name)[1].lower()
        base_meta = {
            "file_name": file_name,
            "total_pages": doc.total_pages,
            "file_hash": file_hash,
            "source_path": file_path,
            "file_type": ext,
        }

        if ext == '.pdf' and doc.total_pages > 0:
            chunks_with_pages = split_text_into_chunks_with_pages(doc.full_text)
            if not chunks_with_pages:
                return
            chunks = [c[0] for c in chunks_with_pages]
            per_chunk_meta = [{"page": c[1]} for c in chunks_with_pages]
        else:
            chunks = split_text_into_chunks(doc.full_text)
            if not chunks:
                return
            per_chunk_meta = None

        # 上下文富化（仅展示用，不参与嵌入）
        sheet = list(doc.table_headers.keys())[0] if doc.table_headers else ""
        header = doc.table_headers.get(sheet, "") if sheet else ""
        display_chunks = enrich_chunks(chunks, doc.file_name, ext,
                                      sheet_name=sheet, table_header=header,
                                      per_chunk_meta=per_chunk_meta)

        try:
            count = rag_engine.index_document(
                chunks, metadata=base_meta, per_chunk_meta=per_chunk_meta,
                display_texts=display_chunks,
            )
            logger.info("已自动入库: %s (%d 个片段)", file_name, count)
        except RuntimeError as exc:
            logger.warning("自动入库失败: %s — %s", file_name, exc)

    return handle_file

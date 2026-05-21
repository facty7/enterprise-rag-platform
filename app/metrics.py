"""Prometheus metrics helpers."""
import time
from contextlib import contextmanager


try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
except Exception:  # pragma: no cover - optional dependency fallback
    Counter = Gauge = Histogram = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    def generate_latest():
        return b""


if Counter:
    CHAT_REQUESTS = Counter("rag_chat_requests_total", "Chat requests", ["mode", "actual_mode", "status"])
    CACHE_EVENTS = Counter("rag_cache_events_total", "Redis cache events", ["event"])
    CHAT_LATENCY = Histogram(
        "rag_chat_latency_seconds",
        "Chat request latency",
        ["mode", "actual_mode"],
        buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 60, 120),
    )
    RETRIEVAL_PHASE = Histogram(
        "rag_retrieval_phase_ms",
        "Retrieval phase latency in milliseconds",
        ["phase", "mode"],
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 3000, 10000),
    )
    INDEXED_DOCS = Gauge("rag_indexed_documents", "Indexed document count", ["collection"])
else:
    CHAT_REQUESTS = CACHE_EVENTS = CHAT_LATENCY = RETRIEVAL_PHASE = INDEXED_DOCS = None


def observe_chat(mode: str, actual_mode: str, status: str, seconds: float):
    if not Counter:
        return
    CHAT_REQUESTS.labels(mode=mode, actual_mode=actual_mode, status=status).inc()
    CHAT_LATENCY.labels(mode=mode, actual_mode=actual_mode).observe(seconds)


def observe_cache(event: str):
    if CACHE_EVENTS:
        CACHE_EVENTS.labels(event=event).inc()


def observe_trace_phases(trace: dict):
    if not RETRIEVAL_PHASE:
        return
    mode = trace.get("mode", "unknown")
    for phase, ms in (trace.get("phase") or {}).items():
        try:
            RETRIEVAL_PHASE.labels(phase=phase, mode=mode).observe(float(ms))
        except Exception:
            pass


def update_indexed_docs(stats: dict):
    if not INDEXED_DOCS:
        return
    INDEXED_DOCS.labels(collection="shared").set(stats.get("shared_chunks", 0))
    INDEXED_DOCS.labels(collection="internal").set(stats.get("internal_chunks", 0))


@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

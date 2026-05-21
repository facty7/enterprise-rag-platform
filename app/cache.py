"""Optional Redis cache helpers for retrieval results."""
import hashlib
import json
import logging
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)
_redis_client = None
_redis_checked = False


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def get_redis():
    """Return a Redis client when cache is enabled and reachable."""
    global _redis_client, _redis_checked
    if not settings.redis_enabled:
        return None
    if _redis_client is not None:
        return _redis_client
    if _redis_checked:
        return None
    _redis_checked = True
    try:
        import redis
        client = redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        _redis_client = client
        logger.info("Redis cache ready: %s", settings.redis_url)
        return _redis_client
    except Exception as exc:
        logger.warning("Redis cache disabled: %s", exc)
        return None


def make_cache_key(namespace: str, payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"rag:{namespace}:{digest}"


def get_json_cache(key: str) -> Optional[dict]:
    client = get_redis()
    if client is None:
        return None
    try:
        value = client.get(key)
        return json.loads(value) if value else None
    except Exception as exc:
        logger.debug("Redis get failed: %s", exc)
        return None


def set_json_cache(key: str, value: dict, ttl: int | None = None) -> bool:
    client = get_redis()
    if client is None:
        return False
    try:
        data = json.dumps(value, ensure_ascii=False, default=_json_default)
        client.setex(key, ttl or settings.cache_ttl_seconds, data)
        return True
    except Exception as exc:
        logger.debug("Redis set failed: %s", exc)
        return False

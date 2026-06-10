"""Cache + rate-limit abstraction with an in-memory default and optional Redis.

Local runs use the in-memory backend (no service required). Setting ``REDIS_URL``
switches to Redis for cross-process rate limiting and result caching. Both backends
implement the same :class:`Cache` protocol so call sites never branch on the backend.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Any, Protocol


class Cache(Protocol):
    """Minimal cache + sliding-window rate limiter interface."""

    def get(self, key: str) -> Any | None:
        """Return a cached value or ``None``."""
        ...

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value with an optional TTL (seconds)."""
        ...

    def hits_in_window(self, key: str, window_seconds: float) -> int:
        """Record an event for ``key`` and return the count within the window."""
        ...


class InMemoryCache:
    """Process-local cache + rate limiter (default backend)."""

    def __init__(self) -> None:
        """Create empty stores."""
        self._store: dict[str, tuple[float | None, Any]] = {}
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def get(self, key: str) -> Any | None:
        """Return a non-expired cached value or ``None``."""
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value with optional TTL."""
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (expires_at, value)

    def hits_in_window(self, key: str, window_seconds: float) -> int:
        """Append now() and return the number of events within the window."""
        now = time.time()
        bucket = self._events[key]
        bucket.append(now)
        while bucket and now - bucket[0] > window_seconds:
            bucket.popleft()
        return len(bucket)


class RedisCache:  # pragma: no cover - exercised only when REDIS_URL is configured
    """Redis-backed cache + rate limiter."""

    def __init__(self, url: str) -> None:
        """Connect to Redis at ``url``."""
        import json

        import redis

        self._json = json
        self._client = redis.Redis.from_url(url)

    def get(self, key: str) -> Any | None:
        """Return a cached value or ``None``."""
        raw = self._client.get(key)
        return None if raw is None else self._json.loads(raw)

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value with an optional TTL."""
        payload = self._json.dumps(value)
        if ttl:
            self._client.setex(key, int(ttl), payload)
        else:
            self._client.set(key, payload)

    def hits_in_window(self, key: str, window_seconds: float) -> int:
        """Record an event and return the count within the sliding window."""
        now = time.time()
        member = f"{now}"
        pipe = self._client.pipeline()
        pipe.zadd(key, {member: now})
        pipe.zremrangebyscore(key, 0, now - window_seconds)
        pipe.zcard(key)
        pipe.expire(key, int(window_seconds) + 1)
        return int(pipe.execute()[2])


_CACHE: Cache | None = None


def get_cache() -> Cache:
    """Return the process-wide cache backend (Redis if configured, else in-memory)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    url = os.environ.get("REDIS_URL")
    if url:
        try:
            _CACHE = RedisCache(url)
            return _CACHE
        except Exception:
            pass
    _CACHE = InMemoryCache()
    return _CACHE


def reset_cache() -> None:
    """Reset the cached backend (used by tests)."""
    global _CACHE
    _CACHE = None

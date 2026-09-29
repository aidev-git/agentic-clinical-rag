"""Shared HTTP plumbing for the three literature source clients:
disk cache, retry-with-backoff, and a simple per-source rate limiter."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

import httpx
from diskcache import Cache

from agentic_clinical_rag.config import get_settings

_cache: Cache | None = None


def _get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache(get_settings().cache_dir)
    return _cache


def _cache_key(source: str, url: str, params: dict[str, Any]) -> str:
    raw = json.dumps({"source": source, "url": url, "params": params}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RateLimiter:
    """Blocking token-bucket-ish limiter: at most `calls_per_second` calls/sec."""

    def __init__(self, calls_per_second: float) -> None:
        self._min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        if self._min_interval == 0.0:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


def fetch(
    source: str,
    url: str,
    params: dict[str, Any],
    limiter: RateLimiter,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    max_attempts: int = 3,
    use_cache: bool = True,
) -> str:
    """GET `url` with retries/backoff/rate-limiting/caching, returning the response body."""
    cache = _get_cache()
    ttl = get_settings().cache_ttl_seconds
    key = _cache_key(source, url, params)

    if use_cache and ttl > 0:
        cached = cache.get(key)
        if cached is not None:
            return cached

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        limiter.wait()
        try:
            response = httpx.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(f"{source}: HTTP {response.status_code}")
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            body = response.text
            if use_cache and ttl > 0:
                cache.set(key, body, expire=ttl)
            return body
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = exc
            time.sleep(2**attempt)

    raise RuntimeError(f"{source}: request failed after {max_attempts} attempts") from last_error

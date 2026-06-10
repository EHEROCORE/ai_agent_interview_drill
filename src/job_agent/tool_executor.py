"""Resilient tool execution: timeout + bounded retry + fallback.

Wraps synchronous tool callables so a slow or flaky tool cannot wedge the pipeline.
Timeout uses a worker thread; retries are bounded; an optional fallback supplies a
safe default instead of raising.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable, TypeVar

T = TypeVar("T")


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails after exhausting retries (and no fallback)."""


def execute(
    fn: Callable[[], T],
    *,
    timeout: float | None = None,
    retries: int = 0,
    fallback: Callable[[], T] | None = None,
) -> T:
    """Run ``fn`` with optional timeout, ``retries`` extra attempts, and ``fallback``."""
    attempts = retries + 1
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            if timeout is None:
                return fn()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(fn).result(timeout=timeout)
        except FuturesTimeout as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc
    if fallback is not None:
        return fallback()
    raise ToolExecutionError(str(last_exc)) from last_exc

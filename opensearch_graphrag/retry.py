"""Retry decorator for Ollama HTTP calls with exponential backoff."""

from __future__ import annotations

import functools
import logging
import time

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)


def with_retry(max_retries: int = 2, backoff_base: float = 1.0):
    """Decorator: retry on transient httpx errors with exponential backoff.

    Usage::

        @with_retry(max_retries=2, backoff_base=1.0)
        def call_ollama(...):
            ...
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except _RETRYABLE as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = backoff_base * (2**attempt)
                        logger.warning(
                            "Retry %d/%d for %s after %s (wait %.1fs)",
                            attempt + 1,
                            max_retries,
                            fn.__name__,
                            type(exc).__name__,
                            delay,
                        )
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator

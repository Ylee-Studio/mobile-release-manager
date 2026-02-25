"""Retry helpers for external tool calls."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 2.0


def call_with_retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= policy.max_attempts:
                break
            if on_retry:
                on_retry(attempt, exc)
            delay = min(policy.base_delay_seconds * (2 ** (attempt - 1)), policy.max_delay_seconds)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc

"""Source-frame age, independent of when a GPU operation finally returns."""
from __future__ import annotations
import time

MAX_DEPTH_AGE_S = 0.25


def depth_is_current(captured_at, now=None):
    if now is None:
        now = time.perf_counter()
    return captured_at is not None and 0 <= now-captured_at <= MAX_DEPTH_AGE_S

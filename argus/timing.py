"""Scalar-only fast-loop timing; no frames or unbounded sample buffers."""
from __future__ import annotations

import time


class TickTiming:
    """Measure consecutive stages, including time descheduled by the OS/GIL.

    The start interval reveals scheduling gaps that work duration alone misses.
    A rejected/missing pair still emits a tick, with depth_ok=False.
    """

    def __init__(self, previous_start=None, clock=time.perf_counter):
        self.clock = clock
        self.start = self.last = clock()
        self.values = {"interval_ms": None if previous_start is None else
                       (self.start - previous_start) * 1000}

    def mark(self, stage):
        now = self.clock()
        self.values[stage + "_ms"] = (now - self.last) * 1000
        self.last = now

    def finish(self, depth_ok=False, **fields):
        return dict(self.values, tick_ms=(self.clock() - self.start) * 1000,
                    depth_ok=depth_ok, **fields)


def heartbeat_gaps(times, begin, end, previous=None, limit_s=0.25):
    """Observe completed AND still-open outages within a measurement window.

    Boundaries are censored observations, not fabricated depth heartbeats.
    If previous is unknown the first interval starts at begin.
    """
    ordered = sorted(t for t in times if begin <= t <= end)
    last = begin if previous is None else previous
    gaps = []
    for t in ordered:
        if t-last > limit_s:
            gaps.append(dict(start=last, end=t, gap_s=t-last, open=False))
        last = t
    if end-last > limit_s:
        gaps.append(dict(start=last, end=end, gap_s=end-last, open=True))
    return gaps

"""Bounded arrival-time pairing for unsynchronised stereo cameras."""
from __future__ import annotations


def select_pair(left, right, consumed, now, max_skew_ms, max_age_s=0.1):
    """Newest gate-compatible pair, advancing BOTH sequences; otherwise None.

    Entries are (frame, monotonic arrival, sequence). Age applies to both
    frames. Arrival skew is not a measurement of exposure synchronisation.
    Histories are bounded by the camera reader, not buffered here.
    """
    candidates = [(l, r) for l in left for r in right
                  if l[2] > consumed[0] and r[2] > consumed[1]
                  and 0 <= now-l[1] <= max_age_s and 0 <= now-r[1] <= max_age_s
                  and (max_skew_ms <= 0 or abs(l[1]-r[1])*1000 <= max_skew_ms)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (min(p[0][1], p[1][1]),
                                         -abs(p[0][1]-p[1][1])))

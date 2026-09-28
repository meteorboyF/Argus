"""Corridor navigation: which way is free, from the calibrated depth map.

This is local guidance, not route planning. It answers "can I keep walking,
and if not, which way is open" from geometry alone, so it is deterministic and
runs inside the fast loop budget (a few hundred microseconds).

Method: the frame is split into `bins` vertical strips. For each strip the
robust range (a low percentile of the finite depths in the walking band) is
the distance to the nearest thing in that direction; the strip's bearing comes
from the camera intrinsics. The widest run of strips whose range exceeds
`clear_distance_m` is the free corridor, and its centre bearing is the
suggested heading. A corridor narrower than `min_corridor_deg` counts as
blocked.

Spoken guidance is rate-limited and only changes when the recommendation
changes, so the user hears "path clear" once, not every tick.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .config import NavigationConfig


@dataclass
class Guidance:
    corridor: list[tuple[float, float]]   # (bearing_deg, range_m) per bin, left -> right
    heading_deg: float | None             # suggested bearing, None when blocked
    width_deg: float                      # angular width of the free corridor
    status: str                           # "clear" | "veer_left" | "veer_right" | "blocked"
    message: str


class Navigator:
    def __init__(self, cfg: NavigationConfig, fx: float, cx: float):
        self.cfg = cfg
        self.fx, self.cx = fx, cx
        self._last_status = ""
        self._last_spoken = 0.0

    def bearings(self, width: int) -> np.ndarray:
        edges = np.linspace(0, width, self.cfg.bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return np.degrees(np.arctan((centers - self.cx) / self.fx))

    def evaluate(self, depth_m: np.ndarray) -> Guidance:
        h, w = depth_m.shape[:2]
        band = depth_m[int(h * self.cfg.band[0]):int(h * self.cfg.band[1]), :]
        edges = np.linspace(0, w, self.cfg.bins + 1).astype(int)
        bearings = self.bearings(w)
        ranges = np.full(self.cfg.bins, np.nan)
        for i in range(self.cfg.bins):
            strip = band[:, edges[i]:edges[i + 1]]
            vals = strip[np.isfinite(strip)]
            if vals.size >= self.cfg.min_pixels_per_bin:
                ranges[i] = np.percentile(vals, self.cfg.percentile)
            elif vals.size == 0 and strip.size:
                # Nothing matched at all: open sky/far wall or textureless. Treat
                # as unknown, never as free.
                ranges[i] = np.nan
        free = np.isfinite(ranges) & (ranges >= self.cfg.clear_distance_m)

        # Widest free run, preferring the one closest to straight ahead.
        best = (0, None)  # (length, (start, end))
        i = 0
        while i < self.cfg.bins:
            if free[i]:
                j = i
                while j + 1 < self.cfg.bins and free[j + 1]:
                    j += 1
                length = j - i + 1
                center = abs(bearings[i:j + 1].mean())
                key = (length, -center)
                if best[1] is None or key > (best[0], -abs(bearings[best[1][0]:best[1][1] + 1].mean())):
                    best = (length, (i, j))
                i = j + 1
            else:
                i += 1

        bin_deg = float(bearings[1] - bearings[0]) if self.cfg.bins > 1 else 0.0
        corridor = list(zip(bearings.tolist(), ranges.tolist()))
        if best[1] is None or best[0] * bin_deg < self.cfg.min_corridor_deg:
            return Guidance(corridor, None, 0.0, "blocked", "Path blocked. Stop.")
        s, e = best[1]
        heading = float(bearings[s:e + 1].mean())
        width = best[0] * bin_deg
        if abs(heading) <= self.cfg.straight_deg:
            return Guidance(corridor, heading, width, "clear", "Path clear ahead.")
        side = "right" if heading > 0 else "left"
        strength = "slightly " if abs(heading) < 2 * self.cfg.straight_deg else ""
        return Guidance(corridor, heading, width, f"veer_{side}",
                        f"Veer {strength}{side}.")

    def spoken(self, guidance: Guidance, now: float | None = None) -> str | None:
        """Return the phrase to speak now, or None (rate-limited, on change)."""
        now = time.monotonic() if now is None else now
        changed = guidance.status != self._last_status
        due = now - self._last_spoken >= self.cfg.repeat_s
        if not self.cfg.speak or not (changed or (due and guidance.status != "clear")):
            return None
        self._last_status = guidance.status
        self._last_spoken = now
        return guidance.message

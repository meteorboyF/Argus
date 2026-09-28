"""Non-ML geometric safety reflex (the fast loop).

Runs continuously and independently of the agent. Looks only at the depth map
and applies simple, deterministic geometric rules — no machine learning, so its
behaviour is predictable and always available, even while the slow loop is busy.

Detects:
  - obstacles ahead within warn / danger distance
  - floor drop-offs (steps down, kerbs, holes) in the path region
  - fast approaches (an incoming vehicle, bicycle or person) by tracking the
    range of each path zone over time and computing time-to-collision
Returns a SafetyState the orchestrator can voice immediately.

Robustness notes (why this isn't a raw argmin):
  - The obstacle distance is a low PERCENTILE of the valid depths in the path
    ROI, not the single minimum pixel — SGBM speckle makes single-pixel minima
    fire false DANGER constantly.
  - SGBM cannot compute disparity in the left `num_disparities`-wide border, so
    drop-off statistics only look at the central columns of the bottom strip.
  - Drop-off detection is debounced over consecutive ticks.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .config import SafetyConfig


class Level(IntEnum):
    CLEAR = 0
    WARN = 1
    DANGER = 2


@dataclass
class SafetyState:
    level: Level
    min_distance_m: float
    direction: str          # "left" | "center" | "right"
    drop_detected: bool
    message: str            # short spoken phrase, or "" when clear
    approach_direction: str | None = None   # zone with the fastest closing
    closing_speed_mps: float = 0.0
    ttc_s: float = float("inf")


ZONES = ("left", "center", "right")


def _direction_of(col: int, width: int) -> str:
    if col < width * 0.35:
        return "left"
    if col > width * 0.65:
        return "right"
    return "center"


class SafetyReflex:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self._drop_ticks = 0  # consecutive ticks the drop signature was present
        # (timestamp, range) history per path zone for the approach rule.
        self._history: dict[str, collections.deque] = {
            zone: collections.deque(maxlen=64) for zone in ZONES}

    # ------------------------------------------------------------ approach
    def _zone_ranges(self, depth_m: np.ndarray) -> dict[str, float]:
        """Robust range per left/center/right zone in the eye-level band.

        The floor is always the nearest surface in the lower path ROI, so an
        approaching vehicle 10 m away would be invisible there. Objects that
        can hit a walking person appear in the middle band of the frame.
        """
        h = depth_m.shape[0]
        roi = depth_m[int(h * self.cfg.approach_band[0]):int(h * self.cfg.approach_band[1]), :]
        finite = np.isfinite(roi)
        w = roi.shape[1]
        bounds = {"left": (0, int(w * 0.35)), "center": (int(w * 0.35), int(w * 0.65)),
                  "right": (int(w * 0.65), w)}
        out = {}
        for zone, (a, b) in bounds.items():
            vals = roi[:, a:b][finite[:, a:b]]
            vals = vals[vals <= self.cfg.approach_max_range_m]
            out[zone] = (float(np.percentile(vals, self.cfg.obstacle_percentile))
                         if vals.size >= max(20, self.cfg.min_valid_pixels // 3)
                         else float("nan"))
        return out

    def _closing(self, zone: str, now: float) -> tuple[float, float]:
        """(closing speed m/s, current range) from a linear fit of recent ranges."""
        hist = [(t, r) for t, r in self._history[zone] if now - t <= self.cfg.approach_window_s]
        if len(hist) < self.cfg.approach_min_samples:
            return 0.0, float("nan")
        t = np.array([h[0] for h in hist]) - now
        r = np.array([h[1] for h in hist])
        if t.ptp() < 1e-3:
            return 0.0, float(r[-1])
        slope, intercept = np.polyfit(t, r, 1)   # metres per second, + = receding
        return float(-slope), float(intercept)

    def approach(self, zone_ranges: dict[str, float], now: float
                 ) -> tuple[str | None, float, float]:
        """Fastest-closing zone -> (zone, closing speed, time-to-collision)."""
        best = (None, 0.0, float("inf"))
        for zone, rng in zone_ranges.items():
            if np.isfinite(rng):
                self._history[zone].append((now, rng))
            speed, current = self._closing(zone, now)
            if speed < self.cfg.approach_min_speed_mps or not np.isfinite(current) or current <= 0:
                continue
            ttc = current / speed
            if ttc < best[2]:
                best = (zone, speed, ttc)
        return best

    def evaluate(self, depth_m: np.ndarray, now: float | None = None) -> SafetyState:
        now = time.monotonic() if now is None else now
        h, w = depth_m.shape[:2]
        # Path region = lower portion of the frame (where the ground/obstacles are).
        roi_top = int(h * (1.0 - self.cfg.roi_bottom_fraction))
        roi = depth_m[roi_top:, :]

        finite = np.isfinite(roi)
        n_valid = int(finite.sum())
        if n_valid < self.cfg.min_valid_pixels:
            # Not enough signal to judge — stay quiet rather than guess.
            return SafetyState(Level.CLEAR, float("inf"), "center", False, "")

        valid_depths = roi[finite]
        min_dist = float(np.percentile(valid_depths, self.cfg.obstacle_percentile))

        # Direction: column of the nearest valid region (use the same percentile
        # cutoff so the direction matches the distance we report).
        near_mask = finite & (roi <= min_dist * 1.1)
        cols = np.where(near_mask.any(axis=0))[0]
        direction = _direction_of(int(np.median(cols)) if cols.size else w // 2, w)

        # Floor drop-off: in the near-bottom strip the floor should return valid,
        # close depths. If most central pixels are invalid OR far (> drop_far_m),
        # the ground has fallen away (step down / kerb / hole).
        strip = depth_m[int(h * 0.85):, int(w * 0.2):int(w * 0.8)]
        if strip.size:
            bad = ~np.isfinite(strip) | (strip > self.cfg.drop_far_m)
            drop_now = bool(bad.mean() > self.cfg.floor_drop_invalid_fraction)
        else:
            drop_now = False
        self._drop_ticks = self._drop_ticks + 1 if drop_now else 0
        drop = self._drop_ticks >= self.cfg.drop_consecutive_ticks

        level = Level.CLEAR
        msg = ""
        if min_dist <= self.cfg.danger_distance_m:
            level = Level.DANGER
            msg = f"Stop. Obstacle very close on your {direction}."
        elif min_dist <= self.cfg.warn_distance_m:
            level = Level.WARN
            msg = f"Careful, obstacle ahead on your {direction}."

        # Something closing fast (vehicle, cyclist, runner) is dangerous well
        # before it reaches the static distance thresholds.
        zone, speed, ttc = self.approach(self._zone_ranges(depth_m), now)
        if zone is not None and ttc <= self.cfg.ttc_danger_s and level < Level.DANGER:
            level = Level.DANGER
            msg = f"Stop. Something is coming fast from your {zone}."
        elif zone is not None and ttc <= self.cfg.ttc_warn_s and level < Level.WARN:
            level = Level.WARN
            msg = f"Careful, something is approaching on your {zone}."

        if drop:
            level = Level.DANGER
            msg = "Stop. Step down ahead." if not msg else "Stop. Step down and obstacle ahead."

        return SafetyState(level, min_dist, direction, drop, msg,
                           approach_direction=zone, closing_speed_mps=speed, ttc_s=ttc)

"""Non-ML geometric safety reflex (the fast loop).

Runs continuously and independently of the agent. Looks only at the depth map
and applies simple, deterministic geometric rules — no machine learning, so its
behaviour is predictable and always available, even while the slow loop is busy.

Detects:
  - obstacles ahead within warn / danger distance
  - floor drop-offs (steps down, kerbs, holes) in the path region
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

    def evaluate(self, depth_m: np.ndarray) -> SafetyState:
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

        if drop:
            level = Level.DANGER
            msg = "Stop. Step down ahead." if not msg else "Stop. Step down and obstacle ahead."

        return SafetyState(level, min_dist, direction, drop, msg)

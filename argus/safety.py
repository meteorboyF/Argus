"""Non-ML geometric safety reflex (the fast loop).

Runs continuously and independently of the agent. Looks only at the depth map
and applies simple, deterministic geometric rules — no machine learning, so its
behaviour is predictable and always available, even while the slow loop is busy.

Detects:
  - obstacles ahead within warn / danger distance
  - floor drop-offs (steps down, kerbs, holes) in the path region
Returns a SafetyState the orchestrator can voice immediately.
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

    def evaluate(self, depth_m: np.ndarray) -> SafetyState:
        h, w = depth_m.shape[:2]
        # Path region = lower portion of the frame (where the ground/obstacles are).
        roi_top = int(h * (1.0 - self.cfg.roi_bottom_fraction))
        roi = depth_m[roi_top:, :]

        finite = np.isfinite(roi)
        if not finite.any():
            return SafetyState(Level.CLEAR, float("inf"), "center", False, "")

        valid = np.where(finite, roi, np.inf)
        min_idx = np.unravel_index(np.argmin(valid), valid.shape)
        min_dist = float(valid[min_idx])
        direction = _direction_of(min_idx[1], w)

        # Floor drop-off: large patch of "very far / no return" in the near-bottom
        # strip suggests the ground fell away (step down / hole).
        bottom_strip = depth_m[int(h * 0.85):, :]
        drop = bool((~np.isfinite(bottom_strip)).mean() > self.cfg.floor_drop_threshold_m)

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

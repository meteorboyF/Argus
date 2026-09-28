"""Approach / time-to-collision rules over synthetic depth sequences."""
from __future__ import annotations

import numpy as np

from argus.config import SafetyConfig
from argus.safety import Level, SafetyReflex

H, W = 240, 320


def scene(zone_ranges: dict[str, float], background: float = 40.0) -> np.ndarray:
    """Open street: far background, floor at 2 m in the bottom strip, and an
    object at eye level in the requested zone(s)."""
    depth = np.full((H, W), background, dtype=np.float32)
    depth[int(H * 0.8):, :] = 2.0
    bounds = {"left": (0, int(W * 0.35)), "center": (int(W * 0.35), int(W * 0.65)),
              "right": (int(W * 0.65), W)}
    for zone, rng in zone_ranges.items():
        a, b = bounds[zone]
        depth[int(H * 0.35):int(H * 0.65), a:b] = rng
    return depth


def cfg() -> SafetyConfig:
    return SafetyConfig(ttc_warn_s=3.0, ttc_danger_s=1.5, approach_min_speed_mps=0.6,
                        approach_window_s=0.8, approach_min_samples=4)


def test_car_closing_at_5_mps_is_flagged_before_static_distance():
    reflex = SafetyReflex(cfg())
    states = []
    for tick in range(8):
        t = tick * 0.1
        rng = 12.0 - 5.0 * t            # 12 m -> 8.5 m within 0.7 s: TTC 1.7 s
        states.append(reflex.evaluate(scene({"left": rng}), now=t))
    final = states[-1]
    assert final.min_distance_m > 1.5, "static distance rules must not be what fired"
    assert final.level is Level.WARN
    assert final.approach_direction == "left"
    assert abs(final.closing_speed_mps - 5.0) < 0.5
    assert 1.5 < final.ttc_s <= 3.0
    assert "approaching on your left" in final.message


def test_imminent_collision_is_danger():
    reflex = SafetyReflex(cfg())
    state = None
    for tick in range(10):
        t = tick * 0.1
        state = reflex.evaluate(scene({"center": 6.0 - 4.0 * t}), now=t)
    assert state.level is Level.DANGER
    assert state.approach_direction == "center"
    assert state.ttc_s <= 1.5
    assert "coming fast" in state.message


def test_static_far_scene_never_reports_approach():
    reflex = SafetyReflex(cfg())
    for tick in range(10):
        state = reflex.evaluate(scene({"right": 6.0}), now=tick * 0.1)
    assert state.level is Level.CLEAR
    assert state.approach_direction is None


def test_receding_object_is_not_an_approach():
    reflex = SafetyReflex(cfg())
    for tick in range(10):
        state = reflex.evaluate(scene({"center": 3.0 + 2.0 * tick * 0.1}), now=tick * 0.1)
    assert state.approach_direction is None
    assert state.level is Level.CLEAR


def test_slow_walking_approach_is_left_to_distance_rules():
    reflex = SafetyReflex(cfg())
    for tick in range(10):
        state = reflex.evaluate(scene({"center": 5.0 - 0.4 * tick * 0.1}), now=tick * 0.1)
    assert state.approach_direction is None
    assert state.level is Level.CLEAR


def test_history_older_than_window_is_ignored():
    reflex = SafetyReflex(cfg())
    for tick in range(6):
        reflex.evaluate(scene({"left": 10.0 - 5.0 * tick * 0.1}), now=tick * 0.1)
    # A long pause, then a static scene: the old fast closing must not carry over.
    for tick in range(6):
        state = reflex.evaluate(scene({"left": 7.0}), now=5.0 + tick * 0.1)
    assert state.approach_direction is None

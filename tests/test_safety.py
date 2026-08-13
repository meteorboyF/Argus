"""Safety reflex geometry tests.

`SafetyReflex.evaluate` is a pure function of a depth map, so the whole
rule-based safety layer can be tested off-device with synthetic depth — no
cameras, no Jetson. This is the loop REQ-NF06 requires to stay auditable;
these tests are what "auditable" means in practice.

Depth maps here are metres, `np.inf` for "no valid disparity" (what SGBM
returns where it cannot match).
"""
from __future__ import annotations

import numpy as np
import pytest

from argus.config import SafetyConfig
from argus.safety import Level, SafetyReflex

H, W = 240, 320


def depth_filled(value: float) -> np.ndarray:
    return np.full((H, W), value, dtype=np.float32)


@pytest.fixture
def cfg():
    # Explicit rather than inherited so a config default change cannot silently
    # rewrite what these tests assert.
    return SafetyConfig(
        warn_distance_m=1.5,
        danger_distance_m=0.7,
        obstacle_percentile=5.0,
        min_valid_pixels=200,
        floor_drop_invalid_fraction=0.75,
        drop_far_m=3.0,
        drop_consecutive_ticks=3,
        roi_bottom_fraction=0.6,
    )


def test_open_space_is_clear(cfg):
    state = SafetyReflex(cfg).evaluate(depth_filled(5.0))
    assert state.level is Level.CLEAR
    assert state.message == ""


def test_close_surface_is_danger(cfg):
    state = SafetyReflex(cfg).evaluate(depth_filled(0.5))
    assert state.level is Level.DANGER
    assert "Stop" in state.message


def test_mid_range_surface_is_warn(cfg):
    state = SafetyReflex(cfg).evaluate(depth_filled(1.2))
    assert state.level is Level.WARN
    assert "Careful" in state.message


def test_boundaries_are_inclusive(cfg):
    """Exactly at a threshold must trigger it — off-by-one here is a missed warning."""
    reflex = SafetyReflex(cfg)
    assert reflex.evaluate(depth_filled(0.7)).level is Level.DANGER
    assert SafetyReflex(cfg).evaluate(depth_filled(1.5)).level is Level.WARN


def test_too_few_valid_pixels_stays_quiet_rather_than_guessing(cfg):
    """A mostly-invalid depth map means "I can't see", not "the way is clear"."""
    depth = depth_filled(np.inf)
    depth[0:5, 0:5] = 0.3          # a handful of very close pixels, under the floor
    state = SafetyReflex(cfg).evaluate(depth)
    assert state.level is Level.CLEAR
    assert state.message == ""


def test_speckle_noise_does_not_fire_danger(cfg):
    """A percentile, not a raw min — a few bad pixels must not scream "Stop"."""
    rng = np.random.default_rng(0)
    depth = depth_filled(5.0)
    ys = rng.integers(0, H, 40)
    xs = rng.integers(0, W, 40)
    depth[ys, xs] = 0.2            # 40 speckle pixels out of 76 800
    assert SafetyReflex(cfg).evaluate(depth).level is Level.CLEAR


def test_obstacle_direction_is_reported(cfg):
    """Direction must match the side the close surface is actually on."""
    depth = depth_filled(5.0)
    depth[int(H * 0.5):, : int(W * 0.25)] = 0.5     # close wall on the left
    state = SafetyReflex(cfg).evaluate(depth)
    assert state.level is Level.DANGER
    assert state.direction == "left"


def test_drop_off_requires_debounce_then_fires(cfg):
    """Drop-off is debounced over consecutive ticks to reject single-frame dropouts."""
    depth = depth_filled(2.0)
    depth[int(H * 0.85):, :] = np.inf          # floor has fallen away
    reflex = SafetyReflex(cfg)
    for _ in range(cfg.drop_consecutive_ticks - 1):
        assert not reflex.evaluate(depth).drop_detected
    final = reflex.evaluate(depth)
    assert final.drop_detected
    assert final.level is Level.DANGER
    assert "Step down" in final.message


def test_drop_off_debounce_resets_on_a_good_frame(cfg):
    """One clean frame must clear the counter, or noise accumulates into a false alarm."""
    dropped = depth_filled(2.0)
    dropped[int(H * 0.85):, :] = np.inf
    reflex = SafetyReflex(cfg)
    reflex.evaluate(dropped)
    reflex.evaluate(dropped)
    reflex.evaluate(depth_filled(2.0))          # floor is back
    assert not reflex.evaluate(dropped).drop_detected

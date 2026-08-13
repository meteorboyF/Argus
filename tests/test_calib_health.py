"""Calibration drift detection tests.

Runs off-device on synthetic stereo pairs. A correctly rectified pair is
simulated by shifting a textured image horizontally (disparity); drift is
simulated by adding a vertical shift, which is exactly what a knocked camera
produces once the stale rectification is applied.

The property under test: a purely horizontal shift must read as healthy at any
disparity, and a vertical component must be detected — because that is the
difference between depth being right and depth being confidently wrong.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from argus.calib_health import CalibrationMonitor, CalibState
from argus.config import DepthConfig

H, W = 480, 640


def textured_image(seed: int = 0) -> np.ndarray:
    """Blob texture — ORB needs corners, so uniform noise alone is a poor scene."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 40, (H, W), dtype=np.uint8)
    for _ in range(260):
        y, x = rng.integers(20, H - 20), rng.integers(20, W - 20)
        r = int(rng.integers(3, 9))
        img[y - r:y + r, x - r:x + r] = int(rng.integers(150, 255))
    return img


def right_view(img: np.ndarray, disparity: int, dy: int = 0) -> np.ndarray:
    """Build the right-eye view of `img`.

    In a rectified pair the left camera sees a point further right than the
    right camera does, so x_R = x_L - disparity for positive disparity — i.e.
    the content moves LEFT in the right image. `dy` injects the vertical offset
    that a drifted calibration produces (it should be zero when healthy).
    """
    out = np.zeros_like(img)
    dx = disparity
    ys, xs = slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx))
    yd, xd = slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx))
    out[yd, xd] = img[ys, xs]
    return out


@pytest.fixture
def cfg():
    # health_downscale=1 so the measured residual is in the same units as the
    # injected shift and the assertions stay legible.
    return DepthConfig(health_downscale=1, health_min_matches=10,
                       health_max_vertical_px=2.0, health_consecutive=3)


def test_well_rectified_pair_reads_near_zero(cfg):
    """Pure horizontal disparity is what a good calibration looks like."""
    left = textured_image()
    right = right_view(left, disparity=24)
    reading = CalibrationMonitor(cfg).measure(left, right)
    assert reading is not None, "textured scene should produce a verdict"
    assert reading.vertical_px < 1.0


def test_vertical_drift_is_measured(cfg):
    """A knocked camera shows up as vertical offset between matched features."""
    left = textured_image()
    right = right_view(left, disparity=24, dy=6)
    reading = CalibrationMonitor(cfg).measure(left, right)
    assert reading is not None
    assert reading.vertical_px == pytest.approx(6.0, abs=1.0)


@pytest.mark.parametrize("disparity", [8, 24, 48])
def test_healthy_at_any_disparity(cfg, disparity):
    """Near and far scenes must both read healthy — the metric is vertical only."""
    left = textured_image(seed=disparity)
    right = right_view(left, disparity=disparity)
    reading = CalibrationMonitor(cfg).measure(left, right)
    assert reading is not None
    assert reading.vertical_px < 1.0


def test_textureless_scene_gives_no_verdict_rather_than_a_pass(cfg):
    """A blank wall means "cannot judge", which must not clear a drift state."""
    blank = np.full((H, W), 128, dtype=np.uint8)
    assert CalibrationMonitor(cfg).measure(blank, blank) is None


def test_state_requires_sustained_breach_before_flagging(cfg):
    """One bad sample must not condemn the calibration."""
    mon = CalibrationMonitor(cfg)
    left = textured_image()
    bad = right_view(left, disparity=24, dy=8)
    for _ in range(cfg.health_consecutive - 1):
        mon._update_state(mon.measure(left, bad))
        assert mon.state is not CalibState.DEGRADED
    mon._update_state(mon.measure(left, bad))
    assert mon.state is CalibState.DEGRADED


def test_state_recovers_after_sustained_good_samples(cfg):
    """Re-calibrating (or re-aiming) must be able to clear the flag."""
    mon = CalibrationMonitor(cfg)
    left = textured_image()
    bad = right_view(left, disparity=24, dy=8)
    good = right_view(left, disparity=24)
    for _ in range(cfg.health_consecutive):
        mon._update_state(mon.measure(left, bad))
    assert mon.state is CalibState.DEGRADED
    for _ in range(cfg.health_consecutive):
        mon._update_state(mon.measure(left, good))
    assert mon.state is CalibState.OK


def test_submit_is_rate_limited(cfg):
    """Called every frame at 10 Hz, it must only actually sample occasionally."""
    cfg.health_interval_s = 60.0
    mon = CalibrationMonitor(cfg)
    try:
        left = textured_image()
        right = right_view(left, disparity=24)
        assert mon.submit(left, right) is True
        assert not any(mon.submit(left, right) for _ in range(20))
    finally:
        mon.stop()


def test_submit_never_blocks_the_caller(cfg):
    """The fast loop calls this; measurement must happen on the worker thread."""
    cfg.health_interval_s = 0.0
    mon = CalibrationMonitor(cfg)
    try:
        left = textured_image()
        right = right_view(left, disparity=24)
        t0 = time.perf_counter()
        mon.submit(left, right)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.005, f"submit() blocked for {elapsed*1000:.1f} ms"
    finally:
        mon.stop()


def test_worker_reaches_degraded_from_submits_alone(cfg):
    """End-to-end through the async path, not just the internal state machine."""
    cfg.health_interval_s = 0.0
    mon = CalibrationMonitor(cfg)
    try:
        left = textured_image()
        bad = right_view(left, disparity=24, dy=8)
        deadline = time.monotonic() + 5.0
        while mon.state is not CalibState.DEGRADED and time.monotonic() < deadline:
            mon.submit(left, bad)
            time.sleep(0.02)
        assert mon.state is CalibState.DEGRADED
        assert mon.last is not None and mon.last.vertical_px > cfg.health_max_vertical_px
    finally:
        mon.stop()


def test_disabling_the_check_costs_nothing(cfg):
    cfg.health_check = False
    mon = CalibrationMonitor(cfg)
    try:
        assert mon.submit(textured_image(), textured_image()) is False
    finally:
        mon.stop()

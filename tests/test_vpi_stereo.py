"""VPI SGM backend: exact disparity on known shifts (skipped without VPI)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

vpi = pytest.importorskip("vpi")


def _pair(shift, w=480, h=300):
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.random((h, w + 60)) * 255).astype(np.uint8), (5, 5), 1.2)
    # Real stereo convention: a point appears further LEFT in the right image.
    return np.ascontiguousarray(tex[:, :w]), np.ascontiguousarray(tex[:, shift:w + shift])


@pytest.mark.parametrize("shift", [5, 20, 40])
def test_vpi_sgm_recovers_known_disparity(shift):
    from argus.vpi_stereo import VpiSgmMatcher
    L, R = _pair(shift)
    d = VpiSgmMatcher().compute(L, R, 64)
    assert abs(float(np.median(d[d > 0])) - shift) < 0.5
    assert (d > 0).mean() > 0.9


def test_depth_estimator_vpi_backend_full_resolution_units():
    from argus.config import DepthConfig
    from argus.depth import DepthEstimator
    cfg = DepthConfig()
    cfg.backend, cfg.calibration_file, cfg.fast_downscale = "vpi_sgm", "/nonexistent.npz", 2
    est = DepthEstimator(cfg)
    try:
        L, R = _pair(40, w=960, h=600)
        d = est.disparity(cv2.cvtColor(L, cv2.COLOR_GRAY2BGR), cv2.cvtColor(R, cv2.COLOR_GRAY2BGR))
        assert d.shape == (600, 960)
        assert abs(float(np.median(d[d > 0])) - 40) < 1.0     # matched at half res, x2
    finally:
        est.health.stop()

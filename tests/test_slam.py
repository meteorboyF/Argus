"""Stereo visual odometry on a synthetic two-layer textured scene."""
import time

import cv2
import numpy as np

from argus.config import SlamConfig
from argus.slam import StereoVisualOdometry

H, W = 240, 320
FX, FY, CX, CY, B = 300.0, 300.0, W / 2, H / 2, 0.1
Z_TOP, Z_BOTTOM = 2.0, 3.0                 # two fronto-parallel layers
STEP_M = 0.04                              # sideways camera step per frame


def scene_pair(tx_m: float):
    """Left/right views of a scene with two textured layers at different
    depths, seen from a camera translated sideways by `tx_m`.

    Two depths matter: on a single plane a sideways translation and a small yaw
    are indistinguishable, so PnP cannot recover scale. With two layers the
    image shift differs per layer (tx * fx / Z) and the motion is unique.
    A point ahead of the rig appears further LEFT in the right image, so the
    right view is each layer's window shifted right by its disparity."""
    rng = np.random.default_rng(0)
    left = np.zeros((H, W), dtype=np.uint8)
    right = np.zeros((H, W), dtype=np.uint8)
    half = H // 2
    for rows, z in (((0, half), Z_TOP), ((half, H), Z_BOTTOM)):
        tex = rng.integers(0, 255, (rows[1] - rows[0], W + 200), dtype=np.uint8)
        tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
        d = int(round(FX * B / z))                 # 15 px / 10 px disparity
        x0 = 100 + int(round(tx_m * FX / z))       # 6 px / 4 px per 0.04 m step
        left[rows[0]:rows[1]] = tex[:, x0:x0 + W]
        right[rows[0]:rows[1]] = tex[:, x0 + d:x0 + d + W]
    return (cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(right, cv2.COLOR_GRAY2BGR))


def wait_processed(vo, n_before, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if len(vo._proc_times) > n_before:
            return
        time.sleep(0.01)
    raise AssertionError("slam worker did not process the frame")


def test_sideways_motion_is_recovered_with_metric_scale():
    vo = StereoVisualOdometry(SlamConfig(min_inliers=10), FX, FY, CX, CY, B)
    try:
        steps = 3
        for i in range(steps + 1):
            l, r = scene_pair(i * STEP_M)
            vo.submit(l, r, ts=i * 0.1)
            wait_processed(vo, i)
        st = vo.state
        assert st.tracking
        assert st.quality > 0.5
        expected = steps * STEP_M                  # 0.12 m sideways
        assert abs(abs(st.x_m) - expected) < 0.02
        assert abs(st.z_m) < 0.03
        assert abs(st.yaw_deg) < 1.0
        assert st.speed_mps > 0.2
        assert len(st.trail) >= steps
    finally:
        vo.stop()


def test_textureless_scene_reports_no_tracking():
    vo = StereoVisualOdometry(SlamConfig(min_inliers=10), FX, FY, CX, CY, B)
    try:
        blank = np.full((H, W, 3), 90, dtype=np.uint8)
        for i in range(3):
            vo.submit(blank, blank, ts=i * 0.1)
            wait_processed(vo, i)
        st = vo.state
        assert not st.tracking
        assert "features" in st.reason or "initialising" in st.reason
    finally:
        vo.stop()

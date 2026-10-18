"""Mount alignment estimator: recover known camera rotations from images and
turn them into the right spoken instruction."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from argus import align

W, H, F = 960, 600, 550.0


def _texture(seed=0, w=2400, h=1600):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (h // 8, w // 8), np.uint8), (w, h),
                     interpolation=cv2.INTER_NEAREST)
    for _ in range(400):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(5, 40)), c[0], -1)
        cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 60)), y + int(rng.integers(5, 60))),
                      c[1], 2)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _view(tex, R, f=F, w=W, h=H):
    """Camera rotated by R looking at a far plane (pure rotation)."""
    K = align.intrinsics(f, w, h)
    th, tw = tex.shape[:2]
    # Plane image is a reference camera with focal F*2 centred on the texture.
    K_tex = align.intrinsics(F * 2.0, tw, th)
    Hm = K @ R @ np.linalg.inv(K_tex)
    return cv2.warpPerspective(tex, Hm, (w, h))


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (10, -3, 2), (-25, 4, -1.5), (3, 1, 0.4)])
def test_angle_round_trip(yaw, pitch, roll):
    R = align.rotation_from_angles(yaw, pitch, roll)
    got = align.angles_from_rotation(R)
    assert np.allclose(got, (yaw, pitch, roll), atol=1e-9)


def test_yaw_right_moves_scene_left():
    """Convention check: a camera turned right sees the straight-ahead point
    on the left of its image."""
    R = align.rotation_from_angles(10, 0, 0)
    d = R @ np.array([0, 0, 1.0])
    assert d[0] < 0


@pytest.mark.parametrize("yaw,pitch,roll", [(12, 0, 0), (-8, 2, 0), (5, -1.5, 1.0), (25, 0, 0)])
def test_recover_relative_rotation_from_images(yaw, pitch, roll):
    tex = _texture()
    left = _view(tex, np.eye(3))
    right = _view(tex, align.rotation_from_angles(yaw, pitch, roll))
    m = align.match_images(left, right)
    assert m is not None
    K = align.intrinsics(F, W, H)
    r = align.rotation_from_homography(m.H, K, K)
    assert abs(r.yaw - yaw) < 0.5
    assert abs(r.pitch - pitch) < 0.5
    assert abs(r.roll - roll) < 0.5


def _pm(vertical=0.0):
    return align.PairMetrics(vertical, 0.0, 0.0, 1.0, 100)


def test_aligned():
    msg, ok = align.advise(align.Rotation(0.3, 0.1, 0.2), _pm(1.0), None, None)
    assert ok and "Lock" in msg


def test_toe_out_moves_right_camera_inward_without_reference():
    # right camera points 20 deg further right than left: turn it inward (left)
    msg, ok = align.advise(align.Rotation(20, 0, 0), _pm(), None, None)
    assert not ok and msg.startswith("Turn the right camera inward") and "20 degrees" in msg


def test_reference_picks_the_camera_that_is_off():
    # left camera points 18 deg left of the wide camera; right is straight.
    left_w = align.Rotation(-18, 0, 0)
    right_w = align.Rotation(1, 0, 0)
    msg, _ = align.advise(align.Rotation(19, 0, 0), _pm(), left_w, right_w)
    assert msg.startswith("Turn the left camera inward")


def test_pitch_and_roll_words():
    up = align.Rotation(0.2, 0, 0)
    msg, _ = align.advise(up, _pm(vertical=12.0), align.Rotation(0, 0, 0),
                          align.Rotation(0, 2.0, 0))
    assert msg.startswith("Tilt the right camera down")
    msg, _ = align.advise(align.Rotation(0, 0, 2.0), _pm(), align.Rotation(0, 0, 0),
                          align.Rotation(0, 0, 2.0))
    assert msg.startswith("Rotate the right camera anticlockwise")


def test_assess_end_to_end_with_wide_reference():
    tex = _texture(1)
    left = _view(tex, align.rotation_from_angles(-10, 0, 0))   # left turned outward
    right = _view(tex, align.rotation_from_angles(2, 0, 0))
    wide = _view(tex, np.eye(3), w=1920, h=1080, f=F * 2)       # same FOV as stereo
    a = align.assess(left, right, wide, F, baseline_m=0.0)
    assert a.pair is not None and abs(a.pair.yaw - 12) < 0.7
    assert a.left_vs_wide is not None and abs(a.left_vs_wide.yaw + 10) < 1.0
    assert a.advice.startswith("Turn the left camera inward")

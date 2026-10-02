"""On-screen timecode used by `argus doctor --skew-test`.

The decoder must read the timestamp back from a camera's view of the monitor:
perspective, downscaling, blur and noise included, and it must refuse frames
it cannot read rather than return a wrong time.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from argus import timecode as tc


@pytest.mark.parametrize("n", [0, 1, 2, 3, 255, 1000, 40000, 65535])
def test_gray_round_trip(n):
    assert tc.gray_decode(tc.gray_encode(n)) == n


def test_adjacent_times_differ_in_one_data_bit():
    a, b = tc.encode_bits(1234)[:16], tc.encode_bits(1235)[:16]
    assert sum(x != y for x, y in zip(a, b)) == 1


def test_parity_rejects_single_bit_corruption():
    bits = tc.encode_bits(4242)
    assert tc.decode_bits(bits) == 4242
    bits[5] ^= 1
    assert tc.decode_bits(bits) is None


def _camera_view(img, t=0.0, size=(960, 600), seed=0):
    """Warp the screen into a camera view: tilt, shrink, blur, noise."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    W, H = size
    dst = np.float32([[90 + 40 * t, 70], [W - 60, 110], [W - 100, H - 60], [70, H - 90 - 30 * t]])
    M = cv2.getPerspectiveTransform(src, dst)
    view = cv2.warpPerspective(img, M, size, borderValue=(60, 60, 60))
    view = cv2.GaussianBlur(view, (5, 5), 1.2)
    rng = np.random.default_rng(seed)
    noisy = view.astype(np.float32) * 0.8 + 20 + rng.normal(0, 6, view.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("t_ms,tilt", [(0, 0.0), (12345, 0.5), (65535, 1.0), (70001, 0.3)])
def test_decode_from_warped_camera_view(t_ms, tilt):
    view = _camera_view(tc.render(t_ms), t=tilt, seed=t_ms % 97)
    assert tc.decode_frame(view) == t_ms % 65536


def test_unreadable_frame_returns_none():
    assert tc.decode_frame(np.full((600, 960, 3), 128, np.uint8)) is None


def test_wrap_diff():
    assert tc.wrap_diff_ms(5, 65530) == 11
    assert tc.wrap_diff_ms(65530, 5) == -11
    assert tc.wrap_diff_ms(100, 90) == 10

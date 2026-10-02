"""Pure parts of `argus doctor`: skew pairing, statistics and role binding."""
from __future__ import annotations

import numpy as np

from argus.cameras import VideoNode
from argus.config import ArgusConfig
from argus.doctor import assign_roles, pair_skews_ms, skew_stats


def test_pair_skews_constant_phase_offset():
    left = [i / 30 for i in range(30)]
    right = [i / 30 + 0.004 for i in range(30)]
    s = pair_skews_ms(left, right)
    assert s.size == 30
    assert np.allclose(s, 4.0, atol=1e-6)


def test_pair_skews_takes_nearest_frame_either_side():
    # Right is 30 ms late at 30 fps: the nearest right frame is the *previous*
    # one, 3.3 ms early, not the matching index.
    left = [i / 30 for i in range(1, 30)]
    right = [i / 30 + 0.030 for i in range(30)]
    assert np.allclose(pair_skews_ms(left, right), 1000 / 30 - 30, atol=1e-6)


def test_skew_stats_gate_fraction():
    st = skew_stats(np.array([1.0, 2.0, 3.0, 20.0]), gate_ms=12.0)
    assert st["n"] == 4 and st["over_gate_pct"] == 25.0 and st["max_ms"] == 20.0


def test_skew_stats_empty():
    assert skew_stats(np.array([]), 12.0) == {"n": 0}


def _node(i, name, port):
    return VideoNode(index=i, name=name, usb_port=port, link_mbps=5000)


def test_assign_roles_full_rig():
    cfg = ArgusConfig()
    cfg.camera.stereo_name_hint, cfg.camera.wide_name_hint = "B0495", "B0459"
    cfg.camera.left_usb_port, cfg.camera.right_usb_port = "2-1.1", "2-1.2"
    nodes = [_node(4, "Arducam B0459", "2-1.4"), _node(2, "Arducam B0495", "2-1.2"),
             _node(0, "Arducam B0495", "2-1.1")]
    roles = assign_roles(nodes, cfg)
    assert roles["left"].usb_port == "2-1.1"
    assert roles["right"].usb_port == "2-1.2"
    assert roles["wide"].usb_port == "2-1.4"


def test_assign_roles_lone_stereo_camera_named_by_port():
    cfg = ArgusConfig()
    cfg.camera.stereo_name_hint = "B0495"
    cfg.camera.left_usb_port, cfg.camera.right_usb_port = "2-1.1", "2-1.2"
    roles = assign_roles([_node(0, "Arducam B0495", "1-2.2")], cfg)
    assert list(roles) == ["right"]

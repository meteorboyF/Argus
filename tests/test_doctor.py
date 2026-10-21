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


def test_probe_reopens_a_camera_that_dropped_off_usb(monkeypatch):
    """A camera that falls off USB comes back as a different /dev/video node.
    The probe must notice the stall and reopen it by hub port, not freeze."""
    import time
    import numpy as np
    from argus import doctor

    class FakeCap:
        def __init__(self, frames):
            self.frames = frames
        def isOpened(self):
            return True
        def get(self, _):
            return 0
        def read(self):
            time.sleep(0.005)
            if self.frames > 0:
                self.frames -= 1
                return True, np.zeros((4, 4, 3), np.uint8)
            return False, None
        def release(self):
            pass

    caps = {0: FakeCap(5), 7: FakeCap(10**6)}
    monkeypatch.setattr(doctor, "_open", lambda index, *a: caps[index])
    old = VideoNode(index=0, name="Arducam B0495", usb_port="2-1.1", link_mbps=5000)
    new = VideoNode(index=7, name="Arducam B0495", usb_port="2-1.1", link_mbps=5000)
    monkeypatch.setattr(doctor, "capture_nodes", lambda *a: [new])
    p = doctor.StreamProbe("left", old, 960, 600, 60, "YUYV", stall_s=0.1)
    try:
        deadline = time.time() + 3
        while time.time() < deadline and p.node.index != 7:
            time.sleep(0.02)
        seq_after_reopen = p.seq
        time.sleep(0.1)
        assert p.node.index == 7 and p.stalls == 1
        assert p.seq > seq_after_reopen          # frames flow again
        assert p.snapshot(max_age_s=0.5) is not None
    finally:
        p.close()


def test_snapshot_refuses_stale_frames():
    import time
    import numpy as np
    from argus import doctor
    p = doctor.StreamProbe.__new__(doctor.StreamProbe)
    import threading
    p._lock, p.rotation = threading.Lock(), 0
    p.latest, p.latest_ts = np.zeros((2, 2, 3), np.uint8), time.perf_counter() - 2.0
    assert p.snapshot() is not None
    assert p.snapshot(max_age_s=1.0) is None


def test_focus_score_prefers_sharp_images():
    import cv2
    import numpy as np
    from argus.doctor import focus_score
    rng = np.random.default_rng(0)
    img = np.zeros((400, 600), np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(150, 420)), int(rng.integers(100, 300))
        cv2.putText(img, "Text", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    assert focus_score(img) > 3 * focus_score(blurred)


def test_focus_advice_follows_the_trend():
    from argus.doctor import focus_advice
    assert focus_advice([10, 10, 22, 22], 22) == "That is the sharpest so far."
    assert focus_advice([30, 30, 20, 18], 30) == "Blurrier. Turn back."
    assert focus_advice([10, 10, 13, 14], 40) == "Sharper. Keep going."

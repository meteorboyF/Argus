"""Approaching person/vehicle pipeline: decode, tracking, range, TTC rules."""
from __future__ import annotations

import math

import numpy as np
import pytest

from argus.movers import (ApproachMonitor, Det, RangeKalman, Track, Tracker, box_range,
                          decode, letterbox)

W, H = 960, 600


def _raw_with(boxes_cls_score, n=5460):
    raw = np.zeros((84, n), np.float32)
    for i, ((cx, cy, w, h), cls, s) in enumerate(boxes_cls_score):
        raw[:4, i] = (cx, cy, w, h)
        raw[4 + cls, i] = s
    return raw


def test_letterbox_and_decode_round_trip_to_frame_pixels():
    frame = np.zeros((H, W, 3), np.uint8)
    _, s, px, py = letterbox(frame)
    assert (s, px, py) == (640 / 960, 0, 8)
    # person at frame box (300,150)-(420,450) -> network coordinates
    x1, y1, x2, y2 = 300 * s + px, 150 * s + py, 420 * s + px, 450 * s + py
    raw = _raw_with([(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), 0, 0.9),
                     ((100, 100, 20, 20), 56, 0.95)])          # a chair: not a mover
    dets = decode(raw, s, px, py, frame.shape)
    assert len(dets) == 1 and dets[0].cls == "person"
    assert np.allclose(dets[0].box, [300, 150, 420, 450], atol=0.5)


def test_nms_keeps_one_of_overlapping_boxes():
    raw = _raw_with([((320, 208, 100, 200), 2, 0.9), ((322, 210, 100, 200), 2, 0.8)])
    assert len(decode(raw, 1.0, 0, 0, (416, 640))) == 1


def _det(x, score=0.9, cls="person"):
    return Det(np.array([x, 200, x + 80, 400], np.float32), score, cls)


def test_tracker_confirms_after_three_hits_and_keeps_identity():
    tr = Tracker()
    assert tr.update([_det(100)]) == [] and tr.update([_det(110)]) == []
    out = tr.update([_det(120)])
    assert len(out) == 1
    tid = out[0].id
    for x in (130, 140, 150):
        out = tr.update([_det(x)])
        assert out[0].id == tid


def test_tracker_rescues_low_score_detection():
    tr = Tracker()
    for x in (100, 110, 120):
        tr.update([_det(x)])
    out = tr.update([_det(130, score=0.3)])        # occluded: low score, same place
    assert len(out) == 1 and out[0].hits == 4


def test_box_range_uses_central_half():
    d = np.full((H, W), 8.0, np.float32)           # background
    d[200:400, 300:380] = 4.0                      # the person
    d[200:400, 300:310] = 8.0                      # box edge leaks background
    assert box_range(d, np.array([290, 190, 390, 410])) == pytest.approx(4.0)


def test_kalman_estimates_closing_speed():
    kf = RangeKalman(10.0, 0.0)
    for i in range(1, 20):
        kf.update(10.0 - 3.0 * 0.1 * i + np.random.default_rng(i).normal(0, 0.05), i * 0.1)
    assert kf.closing == pytest.approx(3.0, abs=0.4)


def _scenario(speed, start=12.0, ego=None, steps=40, dt=0.1):
    """A tracked person closing at `speed` m/s; returns (time, alert) list."""
    mon = ApproachMonitor()
    t = Track(1, np.array([440, 200, 520, 400], np.float32), "person", 0.9, hits=3)
    alerts = []
    for i in range(steps):
        r = start - speed * i * dt
        if r < 0.5:
            break
        d = np.full((H, W), 20.0, np.float32)
        d[200:400, 440:520] = r
        for a in mon.update([t], d, W, i * dt, ego_speed=ego):
            alerts.append((i * dt, a))
    return alerts


def test_fast_approach_warns_then_escalates_to_danger():
    alerts = _scenario(speed=4.0)
    levels = [a.level for _, a in alerts]
    assert levels[0] == "WARN" and "DANGER" in levels
    t_warn = alerts[0][0]
    # 12 m at 4 m/s: TTC 3 s at t=0 already; WARN needs Kalman warm-up + 2 frames.
    assert t_warn < 1.0
    first = alerts[0][1]
    assert first.text == "Person. Ahead." and 2.0 < first.ttc_s <= 3.0 + 0.6


def test_walking_pace_closure_is_not_an_approach_without_ego_motion():
    assert _scenario(speed=1.2) == []


def test_ego_motion_is_subtracted():
    # Wearer walks at 1.2 m/s toward a person walking toward them at 2.5 m/s.
    assert _scenario(speed=3.7, ego=1.2) != []
    # Wearer walks at 1.4 m/s toward a standing person: no alert.
    assert _scenario(speed=1.4, ego=1.4) == []


def test_refractory_limits_repeats():
    alerts = _scenario(speed=2.5, start=15.0, steps=40)
    warns = [t for t, a in alerts if a.level == "WARN"]
    assert len(warns) <= 2

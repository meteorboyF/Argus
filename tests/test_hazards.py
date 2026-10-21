"""Corridor hazard engine on ray-cast synthetic depth (M4)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from argus.hazards import (AlarmManager, HazardEngine, HeadMotionGate, Intrinsics,
                           Observation, fit_ground, observe, points_from_depth)

W, H = 480, 300
K = Intrinsics(f=275.0, cx=W / 2, cy=H / 2)       # 960x600 at f=550, halved
CAM_H, PITCH = 1.6, math.radians(25)   # head-worn cameras look down ~20-25 deg


def render(boxes=(), step_at=None, step_depth=0.3, max_range=8.0):
    """Metric z-depth for a camera CAM_H above the floor, pitched down PITCH.
    boxes: (x0, x1, y0, y1, z0, z1) in world metres (x right, y up, z fwd)."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rx, ry = (u - K.cx) / K.f, (v - K.cy) / K.f
    fwd = np.array([0, -math.sin(PITCH), math.cos(PITCH)])
    down = np.array([0, -math.cos(PITCH), -math.sin(PITCH)])
    D = rx[..., None] * np.array([1.0, 0, 0]) + ry[..., None] * down + fwd   # cam z = 1
    o = np.array([0.0, CAM_H, 0.0])
    t = np.full((H, W), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(D[..., 1] < 0, o[1] / -D[..., 1], np.inf)
        if step_at is not None:
            z_hit = o[2] + tg * D[..., 2]
            tlow = np.where(D[..., 1] < 0, (o[1] + step_depth) / -D[..., 1], np.inf)
            tg = np.where(z_hit > step_at, tlow, tg)
        t = np.minimum(t, tg)
        for x0, x1, y0, y1, z0, z1 in boxes:
            lo = (np.array([x0, y0, z0]) - o) / D
            hi = (np.array([x1, y1, z1]) - o) / D
            tmin = np.max(np.minimum(lo, hi), axis=-1)
            tmax = np.min(np.maximum(lo, hi), axis=-1)
            hit = (tmax >= tmin) & (tmin > 0)
            t = np.where(hit & (tmin < t), tmin, t)
    t[t > max_range] = np.nan
    return t.astype(np.float32)


def test_ground_plane_recovers_camera_height():
    P, _ = points_from_depth(render(), K)
    plane = fit_ground(P)
    assert plane is not None
    assert abs(plane.d - CAM_H) < 0.05           # camera sits 1.6 m above the floor
    assert plane.n[1] < -0.9                     # normal points up (-y)


def test_empty_corridor_raises_nothing():
    eng = HazardEngine(K)
    alerts = [a for i in range(10) for a in eng.tick(render(), now=i * 0.1)]
    assert alerts == []
    assert eng.last.valid and math.isinf(eng.last.body_m)


def test_body_obstacle_needs_three_frames_then_warns():
    eng = HazardEngine(K)
    d = render([(-0.3, 0.3, 0.0, 1.0, 2.0, 2.4)])
    assert eng.tick(d, now=0.0) == [] and eng.tick(d, now=0.1) == []
    alerts = eng.tick(d, now=0.2)
    assert [a.text for a in alerts] == ["Careful. Ahead."]
    assert abs(alerts[0].distance_m - 2.0) < 0.2


def test_close_obstacle_is_danger_and_side_is_reported():
    eng = HazardEngine(K)
    d = render([(0.15, 0.6, 0.0, 1.2, 1.0, 1.3)])        # to the right, 1 m
    alerts = [a for i in range(3) for a in eng.tick(d, now=i * 0.1)]
    assert alerts and alerts[0].level == "DANGER" and alerts[0].text == "Stop. Right."


def test_head_level_obstacle_with_nothing_below():
    eng = HazardEngine(K)
    d = render([(-0.4, 0.4, 1.5, 1.9, 1.6, 1.8)])        # sign/branch at head height
    alerts = [a for i in range(3) for a in eng.tick(d, now=i * 0.1)]
    assert alerts and alerts[0].key == "head"
    assert alerts[0].text.startswith(("Stop. Head height.", "Head height."))
    assert math.isinf(eng.last.body_m)                    # body band is clear


def test_drop_off_ahead():
    eng = HazardEngine(K)
    d = render(step_at=1.8, step_depth=0.3)
    alerts = [a for i in range(3) for a in eng.tick(d, now=i * 0.1)]
    assert any(a.key == "drop" for a in alerts)
    assert eng.last.drop_fraction > 0.35


def test_small_step_below_threshold_is_ignored():
    eng = HazardEngine(K)
    d = render(step_at=1.8, step_depth=0.05)
    alerts = [a for i in range(5) for a in eng.tick(d, now=i * 0.1)]
    assert not any(a.key == "drop" for a in alerts)


def test_refractory_and_escalation():
    eng = HazardEngine(K)
    far = render([(-0.3, 0.3, 0.0, 1.0, 2.2, 2.6)])
    near = render([(-0.3, 0.3, 0.0, 1.0, 1.3, 1.7)])
    said = [a.text for i in range(3) for a in eng.tick(far, now=i * 0.1)]
    assert said == ["Careful. Ahead."]
    # Same hazard, 1 s later: refractory, silent.
    assert eng.tick(far, now=1.0) == []
    # 40% closer: escalating, speaks again despite the refractory period.
    assert [a.text for a in eng.tick(near, now=1.1)] == ["Careful. Ahead."]
    # After the refractory period a persisting hazard is repeated.
    assert eng.tick(near, now=1.2) == []
    assert eng.tick(near, now=5.5) != []


def test_hysteresis_keeps_an_active_alarm_just_past_the_raise_distance():
    m = AlarmManager()
    o = Observation(None, body_m=2.3, valid=True)
    for i in range(3):
        m.update(o, i * 0.1)
    assert m.tracks["body"].active
    m.update(Observation(None, body_m=2.8, valid=True), 0.4)      # < 2.5 * 1.2
    assert m.tracks["body"].active
    m.update(Observation(None, body_m=3.2, valid=True), 0.5)      # beyond: cleared
    assert not m.tracks["body"].active


def test_motion_gate_blocks_new_alarms_but_keeps_active_ones():
    m = AlarmManager()
    o = Observation(None, body_m=2.0, valid=True)
    assert all(m.update(o, i * 0.1, motion_gate=True) == [] for i in range(5))
    assert not m.tracks["body"].active
    m.update(o, 1.0)                       # head still: alarm goes active
    assert m.tracks["body"].active
    m.update(Observation(None, body_m=1.0, valid=True), 1.1, motion_gate=True)
    assert m.tracks["body"].active         # kept (and may escalate) while turning


def test_invalid_depth_never_creates_a_hazard():
    m = AlarmManager()
    assert all(m.update(Observation(None, body_m=1.0, valid=False), i * 0.1) == []
               for i in range(5))


def test_head_motion_gate_detects_turning():
    import cv2
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.random((200, 480)) * 255).astype(np.uint8), (5, 5), 1.5)
    f = 550.0 * 320 / 960                                 # focal for a 320 px wide image
    g = HeadMotionGate(f)
    g.update(tex[:, :320], 0.0)
    assert not g.update(tex[:, :320], 0.1)                # still
    # 30 px in 0.1 s at f=183 px is ~93 deg/s: a fast head turn.
    assert g.update(tex[:, 30:350], 0.2) and g.deg_s > 60
    # 2 px in 0.1 s is ~6 deg/s: walking sway, not a turn.
    assert not g.update(tex[:, 32:352], 0.3)

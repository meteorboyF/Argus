"""Corridor hazard engine (M4): ground plane, height bands, drop-offs, alarms.

Deterministic geometry over a calibrated metric depth map; no learned model
decides anything here. Coordinates:

  camera frame   x right, y down, z forward (rectified left camera)
  ground frame   lateral (right), forward, height above the fitted ground

Rules (research basis: Biped NOA / ImpactAlert, Oct 2026 review):
  - Corridor 0.8 m wide x 2.0 m tall ahead of the wearer.
  - Bands: ground (< 0.15 m, ignored as floor), body 0.15-1.4 m,
    head 1.4-2.0 m. A head-level obstacle (cane blind spot) is its own rule.
  - Drop-off: ground more than 0.10 m below the plane in the 1.5-2.5 m zone.
  - Persistence 3 frames, refractory 4 s unless escalating, distance
    hysteresis (clear at 1.2x the raise distance).
  - Head-motion gate: while global optical flow says the head is turning
    fast, no NEW alarm is raised (unsynchronised stereo is unreliable then);
    alarms already active are kept.
  - Alerts are an earcon plus <= 3 words: "Stop. Left." / "Head height." /
    "Drop ahead."
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass
class Intrinsics:
    f: float
    cx: float
    cy: float


def points_from_depth(depth_m: np.ndarray, K: Intrinsics, step: int = 4,
                      max_range: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) camera-frame points and their (N,2) pixel (row, col), subsampled."""
    d = depth_m[::step, ::step]
    rows, cols = np.mgrid[0:depth_m.shape[0]:step, 0:depth_m.shape[1]:step]
    ok = np.isfinite(d) & (d > 0.2) & (d < max_range)
    z = d[ok]
    u, v = cols[ok].astype(np.float32), rows[ok].astype(np.float32)
    x = (u - K.cx) * z / K.f
    y = (v - K.cy) * z / K.f
    return np.stack([x, y, z], 1), np.stack([v, u], 1)


@dataclass
class Plane:
    n: np.ndarray        # unit normal pointing UP (toward the camera side)
    d: float             # n . p + d = height above the plane

    def height(self, P: np.ndarray) -> np.ndarray:
        return P @ self.n + self.d


def fit_ground(P: np.ndarray, rows: np.ndarray | None = None, iters: int = 120,
               tol: float = 0.04, max_tilt_deg: float = 35.0, rng=None) -> Plane | None:
    """RANSAC plane, constrained near horizontal (normal within max_tilt of
    camera 'up' = -y), fitted on the bottom rows of the image: they see the
    floor the wearer stands on. Fitting on all low points let a larger,
    lower floor beyond a step win the vote and hide the drop."""
    if len(P) < 50:
        return None
    rng = rng or np.random.default_rng(0)
    if rows is not None:
        cand = P[rows >= np.percentile(rows, 70)]
    else:
        cand = P[P[:, 1] > np.percentile(P[:, 1], 40)]
    if len(cand) < 30:
        return None
    up = np.array([0.0, -1.0, 0.0])
    cos_max = math.cos(math.radians(max_tilt_deg))
    if len(cand) > 1500:                          # bound the cost: ~1500 points suffice
        cand = cand[rng.choice(len(cand), 1500, replace=False)]
    # All hypotheses at once (vectorised; a Python loop cost ~25 ms/tick).
    idx = rng.integers(0, len(cand), (iters, 3))
    A, B, C = cand[idx[:, 0]], cand[idx[:, 1]], cand[idx[:, 2]]
    N = np.cross(B - A, C - A)
    norms = np.linalg.norm(N, axis=1)
    ok = norms > 1e-6
    N[ok] /= norms[ok, None]
    N[N @ up < 0] *= -1
    ok &= (N @ up) >= cos_max
    if not ok.any():
        return None
    N, A = N[ok], A[ok]
    D = -np.einsum("ij,ij->i", N, A)
    counts = (np.abs(cand @ N.T + D) < tol).sum(axis=0)
    k = int(np.argmax(counts))
    if counts[k] < 30:
        return None
    Q = cand[np.abs(cand @ N[k] + D[k]) < tol]
    # Least-squares refine: smallest-eigenvalue direction of the 3x3 scatter
    # (a full SVD of the N x 3 matrix built an N x N U and cost ~44 ms).
    c = Q.mean(axis=0)
    w, V = np.linalg.eigh((Q - c).T @ (Q - c))
    n = V[:, 0]
    if n @ up < 0:
        n = -n
    return Plane(n / np.linalg.norm(n), float(-n @ c))


def smooth_plane(prev: Plane | None, new: Plane | None, alpha: float = 0.3) -> Plane | None:
    if new is None:
        return prev
    if prev is None:
        return new
    n = (1 - alpha) * prev.n + alpha * new.n
    n /= np.linalg.norm(n)
    return Plane(n, (1 - alpha) * prev.d + alpha * new.d)


def ground_frame(P: np.ndarray, plane: Plane) -> np.ndarray:
    """(N,3) -> (lateral, forward, height) in a frame on the ground plane."""
    up = plane.n
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up, fwd)        # with y down, up x forward points right
    right = -right if right[0] < 0 else right
    return np.stack([P @ right, P @ fwd, plane.height(P)], 1)


# ---------------------------------------------------------------------------
# Per-tick observation
# ---------------------------------------------------------------------------
@dataclass
class CorridorConfig:
    half_width_m: float = 0.4
    max_forward_m: float = 4.0
    floor_max_h: float = 0.15
    body_max_h: float = 1.4
    head_max_h: float = 2.0
    min_points: int = 12
    drop_zone_m: tuple = (1.5, 2.5)
    drop_below_m: float = 0.10
    drop_min_fraction: float = 0.35
    side_bins: tuple = (-0.4, -0.13, 0.13, 0.4)   # left / centre / right


@dataclass
class Observation:
    plane: Plane | None
    body_m: float = math.inf          # nearest body-band obstacle in the corridor
    body_side: str = "ahead"
    head_m: float = math.inf          # nearest head-band obstacle in the corridor
    head_side: str = "ahead"
    drop: bool = False
    drop_fraction: float = 0.0
    valid: bool = False


def _side(lat: np.ndarray, cfg: CorridorConfig) -> str:
    m = float(np.median(lat))
    if m < cfg.side_bins[1]:
        return "left"
    if m > cfg.side_bins[2]:
        return "right"
    return "ahead"


def observe(depth_m: np.ndarray, K: Intrinsics, prev_plane: Plane | None = None,
            cfg: CorridorConfig = CorridorConfig()) -> Observation:
    P, pix = points_from_depth(depth_m, K)
    plane = smooth_plane(prev_plane, fit_ground(P, pix[:, 0]))
    if plane is None or len(P) < 100:
        return Observation(plane, valid=False)
    G = ground_frame(P, plane)
    lat, fwd, h = G[:, 0], G[:, 1], G[:, 2]
    inc = (np.abs(lat) <= cfg.half_width_m) & (fwd > 0.2) & (fwd <= cfg.max_forward_m)
    obs = Observation(plane, valid=True)

    def nearest(mask):
        sel = inc & mask
        if sel.sum() < cfg.min_points:
            return math.inf, "ahead"
        f = fwd[sel]
        near = f <= np.percentile(f, 10) + 0.15        # robust: the near cluster
        return float(np.percentile(f, 5)), _side(lat[sel][near], cfg)

    obs.body_m, obs.body_side = nearest((h > cfg.floor_max_h) & (h <= cfg.body_max_h))
    obs.head_m, obs.head_side = nearest((h > cfg.body_max_h) & (h <= cfg.head_max_h))
    z0, z1 = cfg.drop_zone_m
    zone = inc & (fwd >= z0) & (fwd <= z1) & (h < cfg.floor_max_h)
    if zone.sum() >= cfg.min_points:
        f_zone, h_zone = fwd[zone], h[zone]
        below = h_zone < -cfg.drop_below_m
        # Ground just past a step edge is hidden from a head-height camera,
        # so a gap in the floor along the corridor counts as well: the
        # fraction of 10 cm forward bins in the zone that hold no on-plane
        # ground, plus the fraction of floor points clearly below the plane.
        bins = np.arange(z0, z1 + 1e-6, 0.10)
        on_plane = f_zone[~below]
        hist, _ = np.histogram(on_plane, bins)
        missing = float((hist == 0).mean())
        obs.drop_fraction = max(float(below.mean()), missing) if below.any() else 0.0
        obs.drop = bool(below.sum() >= cfg.min_points // 2
                        and obs.drop_fraction >= cfg.drop_min_fraction)
    return obs


# ---------------------------------------------------------------------------
# Head-motion gate (no IMU)
# ---------------------------------------------------------------------------
class HeadMotionGate:
    """Head rotation rate from global optical flow (no IMU).

    Global shift (phase correlation) on a small grey image, scaled back to the input
    resolution, converted to degrees with the focal length and divided by
    the time since the previous frame. Above `threshold_deg_s` the head is
    turning too fast for unsynchronised stereo (30 deg/s costs ~5 px of
    disparity at 30 fps), so no NEW alarm may start.
    """

    def __init__(self, f_px: float, threshold_deg_s: float = 30.0, size=(160, 100)):
        self.f_px, self.threshold, self.size = f_px, threshold_deg_s, size
        self._prev = None
        self._prev_t = None
        self.deg_s = 0.0

    def update(self, frame: np.ndarray, now: float) -> bool:
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = g.shape[1] / self.size[0]
        g = cv2.resize(g, self.size, interpolation=cv2.INTER_AREA)
        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = g, now
        if prev is None or now <= prev_t:
            return self.moving
        # A head turn is mostly a global image shift: phase correlation
        # measures it robustly (Farneback reported 3.8 px for a true 30 px
        # pan on fine texture).
        win = cv2.createHanningWindow(self.size, cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(prev.astype(np.float32), g.astype(np.float32), win)
        px = math.hypot(dx, dy) * scale if resp > 0.05 else 0.0
        self.deg_s = math.degrees(math.atan2(px, self.f_px)) / (now - prev_t)
        return self.moving

    @property
    def moving(self) -> bool:
        return self.deg_s > self.threshold


# ---------------------------------------------------------------------------
# Alarm hygiene
# ---------------------------------------------------------------------------
@dataclass
class AlarmConfig:
    persist_frames: int = 3
    refractory_s: float = 4.0
    clear_factor: float = 1.2
    warn_m: float = 2.5          # ~0.3 s pipeline + 0.4 s speech + 1.5 s reaction
    danger_m: float = 1.2
    escalate_frac: float = 0.3   # 30% closer counts as escalating


@dataclass
class Alert:
    key: str                     # "body" | "head" | "drop"
    level: str                   # "WARN" | "DANGER"
    text: str                    # <= 3 words
    distance_m: float


@dataclass
class _Track:
    count: int = 0
    active: bool = False
    level: str = ""
    last_said: float = -math.inf
    said_distance: float = math.inf


@dataclass
class AlarmManager:
    cfg: AlarmConfig = field(default_factory=AlarmConfig)
    tracks: dict = field(default_factory=dict)

    @staticmethod
    def phrase(key: str, level: str, side: str) -> str:
        where = {"left": "Left.", "right": "Right.", "ahead": "Ahead."}[side]
        if key == "head":
            return "Stop. Head height." if level == "DANGER" else "Head height. " + where
        if key == "drop":
            return "Stop. Drop ahead." if level == "DANGER" else "Drop ahead."
        return ("Stop. " if level == "DANGER" else "Careful. ") + where

    def update(self, obs: Observation, now: float, motion_gate: bool = False) -> list[Alert]:
        """Alerts to speak this tick (usually none)."""
        out = []
        cand = {
            "body": (obs.body_m, obs.body_side),
            "head": (obs.head_m, obs.head_side),
            "drop": (2.0 if obs.drop else math.inf, "ahead"),
        }
        for key, (dist, side) in cand.items():
            tr = self.tracks.setdefault(key, _Track())
            raise_at = self.cfg.warn_m
            clear_at = self.cfg.warn_m * self.cfg.clear_factor
            present = dist <= (clear_at if tr.active else raise_at)
            if not obs.valid:
                present = tr.active and present   # never invent hazards from no data
            if not present:
                tr.count, tr.active, tr.level = 0, False, ""
                continue
            tr.count += 1
            level = "DANGER" if (dist <= self.cfg.danger_m or key == "drop" and dist <= 2.0
                                 and tr.count >= 2 * self.cfg.persist_frames) else "WARN"
            if key == "head" and dist <= self.cfg.danger_m * 1.5:
                level = "DANGER"   # head strikes give less warning: be earlier
            if not tr.active:
                if tr.count < self.cfg.persist_frames or motion_gate:
                    continue       # not persistent yet / head turning: no NEW alarm
                tr.active = True
            escalating = (level == "DANGER" and tr.level != "DANGER") or \
                dist < tr.said_distance * (1 - self.cfg.escalate_frac)
            if now - tr.last_said >= self.cfg.refractory_s or escalating:
                out.append(Alert(key, level, self.phrase(key, level, side), dist))
                tr.last_said, tr.said_distance = now, dist
            tr.level = level
        out.sort(key=lambda a: (a.level != "DANGER", a.distance_m))
        return out


class HazardEngine:
    """observe() + head-motion gate + alarm hygiene, one call per fast tick."""

    def __init__(self, K: Intrinsics, corridor: CorridorConfig = CorridorConfig(),
                 alarms: AlarmConfig = AlarmConfig()):
        self.K, self.corridor = K, corridor
        self.plane: Plane | None = None
        self.gate = HeadMotionGate(K.f)
        self.alarms = AlarmManager(alarms)
        self.last: Observation | None = None

    def tick(self, depth_m: np.ndarray, left_gray: np.ndarray | None = None,
             now: float | None = None) -> list[Alert]:
        now = time.monotonic() if now is None else now
        moving = self.gate.update(left_gray, now) if left_gray is not None else False
        obs = observe(depth_m, self.K, self.plane, self.corridor)
        if obs.plane is not None:
            self.plane = obs.plane
        self.last = obs
        return self.alarms.update(obs, now, motion_gate=moving)

"""Approaching person / vehicle warnings (M4).

  YOLO11n TensorRT FP16 on the LEFT stereo frame (boxes align with the depth
  map pixel-for-pixel, so no cross-camera projection is needed)
  -> ByteTrack-style two-stage IoU tracker
  -> range per track = median depth of the box's central half
  -> constant-velocity Kalman on range -> closing speed -> time-to-collision
  -> WARN TTC < 3 s, DANGER TTC < 1.5 s, with hysteresis (clear at 1.2x)

Research basis: raw TTC alone was ~51% accurate in Biped's evaluation;
hysteresis plus stable tracking added ~42 points. No torch at runtime: the
engine was exported once with Ultralytics and built with trtexec
(3.3 ms median GPU compute at 640x416).

Ego motion: walking toward a standing person also closes the range. When
visual odometry gives the wearer's forward speed it is subtracted; without
it, a target must close faster than a brisk walk (1.8 m/s) to count as
approaching on its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

MOVER_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
SPOKEN = {"person": "Person", "bicycle": "Bike", "car": "Car", "motorcycle": "Motorbike",
          "bus": "Bus", "truck": "Truck"}


@dataclass
class Det:
    box: np.ndarray          # x1, y1, x2, y2 in frame pixels
    score: float
    cls: str


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, size=(640, 416)):
    h, w = img.shape[:2]
    s = min(size[0] / w, size[1] / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((size[1], size[0], 3), 114, np.uint8)
    px, py = (size[0] - nw) // 2, (size[1] - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # One C++ call for BGR->RGB, HWC->CHW, scale: the numpy version cost ~15 ms.
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, swapRB=True)
    return blob, s, px, py


def decode(raw: np.ndarray, scale: float, px: int, py: int, frame_shape,
           conf: float = 0.25, iou: float = 0.5) -> list[Det]:
    """raw: (84, N) YOLO11 head (cx, cy, w, h, 80 class scores)."""
    cls_scores = raw[4:]
    keep_cls = np.array(sorted(MOVER_CLASSES))
    sub = cls_scores[keep_cls]
    best = sub.argmax(axis=0)
    score = sub.max(axis=0)
    m = score >= conf
    if not m.any():
        return []
    cx, cy, w, h = raw[0, m], raw[1, m], raw[2, m], raw[3, m]
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
    sc, cl = score[m], keep_cls[best[m]]
    idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), sc.tolist(), cl.tolist(), conf, iou)
    out = []
    H, W = frame_shape[:2]
    for i in np.asarray(idx).reshape(-1):
        x, y, bw, bh = boxes[i]
        b = np.array([(x - px) / scale, (y - py) / scale,
                      (x + bw - px) / scale, (y + bh - py) / scale], np.float32)
        b[[0, 2]] = np.clip(b[[0, 2]], 0, W - 1)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, H - 1)
        out.append(Det(b, float(sc[i]), MOVER_CLASSES[int(cl[i])]))
    return out


class MoverDetector:
    def __init__(self, engine_path: str, conf: float = 0.25, size=(640, 416)):
        from .trt_runner import TRTRunner
        self.runner = TRTRunner(engine_path)
        self.conf, self.size = conf, size
        shape = tuple(self.runner.inputs[0]["shape"])
        if shape != (1, 3, size[1], size[0]):
            raise RuntimeError(f"detector engine input {shape} != (1, 3, {size[1]}, {size[0]})")

    def detect(self, frame_bgr: np.ndarray) -> list[Det]:
        blob, s, px, py = letterbox(frame_bgr, self.size)
        raw = next(iter(self.runner.infer({"images": blob}).values()))[0]
        return decode(raw, s, px, py, frame_bgr.shape, self.conf)


# ---------------------------------------------------------------------------
# Tracker (ByteTrack-style: high-score match, then low-score rescue)
# ---------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


@dataclass
class Track:
    id: int
    box: np.ndarray
    cls: str
    score: float
    hits: int = 1
    misses: int = 0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3

    def predict(self) -> np.ndarray:
        return self.box + self.vel


class Tracker:
    def __init__(self, high: float = 0.5, low: float = 0.15, match_iou: float = 0.3,
                 max_misses: int = 5):
        self.high, self.low, self.match_iou, self.max_misses = high, low, match_iou, max_misses
        self.tracks: list[Track] = []
        self._next = 1

    @staticmethod
    def _greedy(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
        pairs = []
        iou = iou.copy()
        while iou.size and iou.max() >= thr:
            i, j = np.unravel_index(np.argmax(iou), iou.shape)
            pairs.append((int(i), int(j)))
            iou[i, :], iou[:, j] = -1, -1
        return pairs

    def update(self, dets: list[Det]) -> list[Track]:
        hi = [d for d in dets if d.score >= self.high]
        lo = [d for d in dets if self.low <= d.score < self.high]
        pred = np.array([t.predict() for t in self.tracks]) if self.tracks else np.zeros((0, 4))
        unmatched_t = list(range(len(self.tracks)))

        def apply(pairs, pool, tracks_idx):
            used = set()
            for ti, di in pairs:
                t, d = self.tracks[tracks_idx[ti]], pool[di]
                t.vel = 0.6 * t.vel + 0.4 * (d.box - t.box)
                t.box, t.score, t.cls = d.box, d.score, d.cls
                t.hits += 1
                t.misses = 0
                used.add(di)
            return used

        if hi and len(unmatched_t):
            m = iou_matrix(pred[unmatched_t], np.array([d.box for d in hi]))
            pairs = self._greedy(m, self.match_iou)
            used_hi = apply(pairs, hi, unmatched_t)
            matched = {unmatched_t[ti] for ti, _ in pairs}
            unmatched_t = [i for i in unmatched_t if i not in matched]
        else:
            used_hi = set()
        if lo and unmatched_t:
            m = iou_matrix(pred[unmatched_t], np.array([d.box for d in lo]))
            pairs = self._greedy(m, 0.5)
            apply(pairs, lo, unmatched_t)
            matched = {unmatched_t[ti] for ti, _ in pairs}
            unmatched_t = [i for i in unmatched_t if i not in matched]
        for i in unmatched_t:
            t = self.tracks[i]
            t.misses += 1
            t.box = t.predict()
        for di, d in enumerate(hi):
            if di not in used_hi:
                self.tracks.append(Track(self._next, d.box.copy(), d.cls, d.score))
                self._next += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.confirmed and t.misses == 0]


# ---------------------------------------------------------------------------
# Range, Kalman, TTC with hysteresis
# ---------------------------------------------------------------------------
def box_range(depth_m: np.ndarray, box: np.ndarray, min_pixels: int = 20) -> float:
    """Median valid depth of the central half of the box (edges hit background)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(x1 + 0.25 * w), int(math.ceil(x2 - 0.25 * w))
    ya, yb = int(y1 + 0.25 * h), int(math.ceil(y2 - 0.25 * h))
    roi = depth_m[max(0, ya):yb, max(0, xa):xb]
    v = roi[np.isfinite(roi) & (roi > 0.2)]
    return float(np.median(v)) if v.size >= min_pixels else math.nan


class RangeKalman:
    """Constant-velocity Kalman on range: state (r, dr/dt)."""

    def __init__(self, r0: float, t0: float, meas_std: float = 0.15, accel_std: float = 1.5):
        self.x = np.array([r0, 0.0])
        self.P = np.diag([meas_std ** 2, 4.0])
        self.t = t0
        self.R = meas_std ** 2
        self.q = accel_std ** 2
        self.n = 1

    def update(self, r: float, t: float) -> None:
        dt = max(1e-3, t - self.t)
        self.t = t
        F = np.array([[1, dt], [0, 1]])
        Q = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if math.isfinite(r):
            H = np.array([1.0, 0.0])
            S = H @ self.P @ H + self.R
            K = self.P @ H / S
            self.x = self.x + K * (r - H @ self.x)
            self.P = (np.eye(2) - np.outer(K, H)) @ self.P
            self.n += 1

    @property
    def range(self) -> float:
        return float(self.x[0])

    @property
    def closing(self) -> float:
        return float(-self.x[1])


@dataclass
class MoverAlert:
    track_id: int
    level: str
    text: str
    ttc_s: float
    range_m: float
    closing_mps: float


@dataclass
class ApproachConfig:
    warn_ttc: float = 3.0
    danger_ttc: float = 1.5
    clear_factor: float = 1.2
    min_closing: float = 0.5        # m/s, ignore drift
    walk_speed: float = 1.8         # without ego motion, must out-pace a brisk walk
    min_updates: int = 4            # Kalman needs a few ranges before trusting speed
    persist: int = 2                # consecutive updates under a threshold
    refractory_s: float = 3.0


class ApproachMonitor:
    def __init__(self, cfg: ApproachConfig = ApproachConfig()):
        self.cfg = cfg
        self.kf: dict[int, RangeKalman] = {}
        self.state: dict[int, dict] = {}

    def update(self, tracks: list[Track], depth_m: np.ndarray, frame_w: int, now: float,
               ego_speed: float | None = None) -> list[MoverAlert]:
        out = []
        live = set()
        for t in tracks:
            live.add(t.id)
            r = box_range(depth_m, t.box)
            kf = self.kf.get(t.id)
            if kf is None:
                if math.isfinite(r):
                    self.kf[t.id] = RangeKalman(r, now)
                    self.state[t.id] = {"level": "", "count": 0, "said": -math.inf}
                continue
            kf.update(r, now)
            st = self.state[t.id]
            closing = kf.closing - (ego_speed or 0.0)
            need = self.cfg.min_closing if ego_speed is not None else self.cfg.walk_speed
            ttc = kf.range / closing if closing > need else math.inf
            if kf.n < self.cfg.min_updates:
                continue
            active = st["level"] != ""
            warn_at = self.cfg.warn_ttc * (self.cfg.clear_factor if active else 1.0)
            if ttc <= self.cfg.danger_ttc:
                level = "DANGER"
            elif ttc <= warn_at:
                level = "WARN"
            else:
                level = ""
            st["count"] = st["count"] + 1 if level else 0
            if not level:
                st["level"] = ""
                continue
            if not active and st["count"] < self.cfg.persist:
                continue
            escalate = level == "DANGER" and st["level"] != "DANGER"
            st["level"] = level
            if escalate or now - st["said"] >= self.cfg.refractory_s:
                cx = (t.box[0] + t.box[2]) / 2
                side = "Left." if cx < frame_w * 0.35 else "Right." if cx > frame_w * 0.65 else "Ahead."
                word = SPOKEN.get(t.cls, "Something")
                text = f"Stop. {word}. {side[:-1]}." if level == "DANGER" else f"{word}. {side}"
                out.append(MoverAlert(t.id, level, text, ttc, kf.range, closing))
                st["said"] = now
        for tid in list(self.kf):
            if tid not in live:
                self.kf.pop(tid, None)
                self.state.pop(tid, None)
        return out

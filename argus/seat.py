"""Seat guidance (M7): "find me a seat", closed loop, no SLAM dependency.

Every 0.5-1 s the seat is re-detected (YOLO11n TensorRT on the left stereo
frame: chair / couch / bench) and spoken guidance is refreshed:

  "Seat, slightly right, 4 steps."            (calibrated: range -> steps)
  "Seat, slightly right."                     (uncalibrated: direction only)
  "Seat in front of you, knee height."        (final approach, < 0.9 m)
  "Turn slowly left..."                       (target lost: search)

Range comes from the median depth of the box's central half; steps assume
0.7 m per step. The corridor safety rules stay active throughout: this module
only produces GUIDANCE-priority speech, which hazards always cut.

The target is "whitelisted" for the corridor rule: during the approach the
seat itself is expected ahead and must not trigger "Careful. Ahead." That
whitelist is the only way the slow loop may configure the fast loop, and it
can never disable drop-off or head-level rules.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

SEAT_CLASSES = {56: "chair", 57: "couch", 13: "bench"}
STEP_M = 0.7


@dataclass
class SeatSighting:
    box: np.ndarray
    score: float
    cls: str


def detect_seats(raw: np.ndarray, scale: float, px: int, py: int, frame_shape,
                 conf: float = 0.35) -> list[SeatSighting]:
    """Decode YOLO11 (84, N) output for seat classes only."""
    import cv2
    keep = np.array(sorted(SEAT_CLASSES))
    sub = raw[4:][keep]
    score = sub.max(axis=0)
    m = score >= conf
    if not m.any():
        return []
    cx, cy, w, h = raw[0, m], raw[1, m], raw[2, m], raw[3, m]
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
    cls = keep[sub.argmax(axis=0)[m]]
    sc = score[m]
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), conf, 0.5)
    H, W = frame_shape[:2]
    out = []
    for i in np.asarray(idx).reshape(-1):
        x, y, bw, bh = boxes[i]
        b = np.array([(x - px) / scale, (y - py) / scale,
                      (x + bw - px) / scale, (y + bh - py) / scale], np.float32)
        b[[0, 2]] = np.clip(b[[0, 2]], 0, W - 1)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, H - 1)
        out.append(SeatSighting(b, float(sc[i]), SEAT_CLASSES[int(cls[i])]))
    return out


def bearing_deg(box: np.ndarray, frame_w: int, f_px: float) -> float:
    cx = (box[0] + box[2]) / 2
    return math.degrees(math.atan2(cx - frame_w / 2, f_px))


def direction_words(bearing: float) -> str:
    a = abs(bearing)
    if a < 6:
        return "straight ahead"
    side = "right" if bearing > 0 else "left"
    if a < 20:
        return f"slightly {side}"
    if a < 45:
        return f"to your {side}"
    return f"far {side}"


def guidance_phrase(bearing: float, range_m: float | None) -> str:
    """Short spoken update; distance only when range is metric (calibrated)."""
    where = direction_words(bearing)
    if range_m is not None and math.isfinite(range_m):
        if range_m < 0.9 and abs(bearing) < 15:
            return "Seat in front of you, knee height."
        steps = max(1, int(round(range_m / STEP_M)))
        return f"Seat, {where}, {steps} step{'s' if steps > 1 else ''}."
    return f"Seat, {where}."


@dataclass
class SeatGuideState:
    phrase: str = ""
    bearing: float = math.nan
    range_m: float = math.nan
    lost_since: float | None = None
    arrived: bool = False


class SeatGuide:
    """Pure decision logic; the orchestrator feeds detections + depth."""

    def __init__(self, f_px: float, lost_after_s: float = 2.0):
        self.f_px = f_px
        self.lost_after_s = lost_after_s
        self.state = SeatGuideState()
        self._last_bearing = 0.0

    def update(self, seats: list[SeatSighting], frame_w: int,
               depth_m: np.ndarray | None, now: float) -> str:
        if not seats:
            if self.state.lost_since is None:
                self.state.lost_since = now
            if now - self.state.lost_since >= self.lost_after_s:
                side = "left" if self._last_bearing < 0 else "right"
                self.state.phrase = f"Turn slowly {side}. I am looking for a seat."
            return self.state.phrase
        self.state.lost_since = None
        # Prefer the closest seat when range is known, else the most central.
        rng = []
        for s in seats:
            r = math.nan
            if depth_m is not None:
                from .movers import box_range
                r = box_range(depth_m, s.box)
            rng.append(r)
        if any(math.isfinite(r) for r in rng):
            i = int(np.nanargmin(rng))
        else:
            i = int(np.argmin([abs(bearing_deg(s.box, frame_w, self.f_px)) for s in seats]))
        b = bearing_deg(seats[i].box, frame_w, self.f_px)
        r = rng[i] if math.isfinite(rng[i]) else None
        self._last_bearing = b
        self.state.bearing, self.state.range_m = b, (r if r is not None else math.nan)
        self.state.phrase = guidance_phrase(b, r)
        self.state.arrived = self.state.phrase.startswith("Seat in front")
        return self.state.phrase


class SeatSession:
    """Runs one guided approach: re-detect every `period_s`, speak changes."""

    def __init__(self, orch, period_s: float = 0.7, timeout_s: float = 90.0):
        self.orch, self.period_s, self.timeout_s = orch, period_s, timeout_s

    def run(self) -> bool:
        from .movers import letterbox
        from .speech import Priority
        from .telemetry import bus
        o = self.orch
        det, _, _ = o._movers if getattr(o, "_movers", None) else (None, None, None)
        if det is None:
            o.speaker.speak("Seat finding is not available right now.")
            return False
        f_px = o.depth.intrinsics()["fx"] if o.depth.calibrated else o.cfg.depth.focal_px
        guide = SeatGuide(f_px)
        o.speaker.speak("Looking for a seat.", Priority.GUIDANCE)
        said, said_at = "", 0.0
        t0 = time.monotonic()
        o.seat_cancel.clear()
        while (time.monotonic() - t0 < self.timeout_s and not o._stop.is_set()
               and not o.seat_cancel.is_set()):
            pair = o.rig.get_stereo_pair()
            frame = None
            if o.depth.calibrated and o.depth.last_rectified is not None:
                frame = o.depth.last_rectified[0]
            elif pair is not None:
                frame = pair.left
            if frame is None:
                time.sleep(0.05)
                continue
            blob, s, px, py = letterbox(frame)
            raw = next(iter(det.runner.infer({"images": blob}).values()))[0]
            seats = detect_seats(raw, s, px, py, frame.shape)
            depth = o.latest_depth() if o.depth.calibrated else None
            if depth is not None and depth.shape[:2] != frame.shape[:2]:
                import cv2
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            phrase = guide.update(seats, frame.shape[1], depth, time.monotonic())
            bus.set(seat=phrase or "searching")
            # Configure down, never disable: the target seat may sit in the
            # corridor without triggering obstacle speech; drop-off and
            # head-level rules are untouched.
            if math.isfinite(guide.state.range_m):
                o.corridor_whitelist = (guide.state.range_m, time.monotonic() + 1.5)
            now = time.monotonic()
            if phrase and (phrase != said or now - said_at > 4.0) and not o.speaker.busy:
                o.speaker.speak(phrase, Priority.GUIDANCE)
                said, said_at = phrase, now
            if guide.state.arrived:
                o.corridor_whitelist = None
                o.speaker.wait_until_idle(timeout=5.0)
                return True
            time.sleep(self.period_s)
        o.corridor_whitelist = None
        o.speaker.speak("I stopped looking for a seat.", Priority.GUIDANCE)
        return False


def is_seat_request(q: str) -> bool:
    import re
    return bool(re.search(r"\b(find|where|get|need|want)\b.*\b(seat|chair|sofa|bench|"
                          r"somewhere to sit|sit down)\b|\bseat me\b|\bi want to sit\b", q,
                          re.IGNORECASE))

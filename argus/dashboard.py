"""On-monitor dashboard: what ARGUS sees, decides and says, live.

One OpenCV window, rendered on its own thread from the telemetry bus. Layout
on a 1920x1080 monitor:

    +------------------+------------------+------------------+
    | LEFT stereo      | RIGHT stereo     | WIDE + detections|
    +------------------+------------------+------------------+
    | DEPTH + corridor | TOP-DOWN map     | STATUS           |
    +------------------+------------------+------------------+
    | conversation: You / ARGUS / safety warnings   | typed question box |
    +------------------------------------------------------------+

Keys: type a question and press Enter to ask ARGUS (works without a
microphone), Backspace edits, Esc clears, `q` (on an empty line) quits.
"""
from __future__ import annotations

import textwrap
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .telemetry import Snapshot, bus

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (18, 18, 18)
PANEL = (32, 32, 32)
FG = (230, 230, 230)
DIM = (140, 140, 140)
GREEN = (80, 220, 80)
AMBER = (0, 190, 255)
RED = (60, 60, 255)
CYAN = (230, 200, 60)
MAGENTA = (200, 80, 230)

KIND_COLOR = {"user": CYAN, "argus": GREEN, "safety": RED, "system": DIM}
KIND_LABEL = {"user": "You", "argus": "ARGUS", "safety": "SAFETY", "system": "sys"}


def _fit(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox `frame` into a w x h tile."""
    out = np.full((h, w, 3), PANEL, dtype=np.uint8)
    if frame is None or frame.size == 0:
        return out
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    scale = min(w / frame.shape[1], h / frame.shape[0])
    rw, rh = max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
    x, y = (w - rw) // 2, (h - rh) // 2
    out[y:y + rh, x:x + rw] = resized
    return out


def _title(tile: np.ndarray, text: str, color=FG) -> None:
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(tile, text, (8, 19), FONT, 0.55, color, 1, cv2.LINE_AA)


def depth_to_color(depth_m: np.ndarray, max_m: float = 5.0) -> np.ndarray:
    """Near = warm, far = cool, invalid = black."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    norm = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m[valid], 0.0, max_m) / max_m
        norm[valid] = (255 - clipped * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def disparity_to_color(disp: np.ndarray) -> np.ndarray:
    valid = disp > 0
    norm = np.zeros(disp.shape, dtype=np.uint8)
    if valid.any():
        hi = max(1.0, float(np.percentile(disp[valid], 98)))
        norm[valid] = np.clip(disp[valid] / hi * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _draw_detections(frame: np.ndarray, dets: list) -> np.ndarray:
    out = frame.copy()
    for det in dets:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), MAGENTA, 3)
        label = f"{det.name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.8, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), MAGENTA, -1)
        cv2.putText(out, label, (x1 + 5, y1 - 6), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _draw_topdown(w: int, h: int, values: dict) -> np.ndarray:
    """Ego-centric top-down view: corridor occupancy plus SLAM trail."""
    tile = np.full((h, w, 3), PANEL, dtype=np.uint8)
    cx, base = w // 2, h - 30
    px_per_m = (h - 60) / 5.0
    for r in (1, 2, 3, 4, 5):
        y = int(base - r * px_per_m)
        cv2.line(tile, (20, y), (w - 20, y), (55, 55, 55), 1)
        cv2.putText(tile, f"{r} m", (24, y - 4), FONT, 0.4, DIM, 1, cv2.LINE_AA)
    corridor = values.get("corridor")   # list of (angle_deg, range_m) per column bin
    if corridor:
        for angle, rng in corridor:
            if not np.isfinite(rng):
                continue
            rad = np.deg2rad(angle)
            r = min(rng, 5.0) * px_per_m
            x = int(cx + np.sin(rad) * r)
            y = int(base - np.cos(rad) * r)
            col = RED if rng < 0.7 else AMBER if rng < 1.5 else GREEN
            cv2.circle(tile, (x, y), 4, col, -1)
    trail = values.get("slam_trail")   # list of (x_m, z_m) in the start frame
    if trail and len(trail) > 1:
        pts = np.array(trail, dtype=np.float32)
        pts -= pts[-1]
        yaw = float(values.get("slam_yaw_deg", 0.0))
        c, s = np.cos(np.deg2rad(-yaw)), np.sin(np.deg2rad(-yaw))
        rot = pts @ np.array([[c, -s], [s, c]], dtype=np.float32).T
        poly = [(int(cx + x * px_per_m), int(base - z * px_per_m)) for x, z in rot]
        cv2.polylines(tile, [np.array(poly, dtype=np.int32)], False, CYAN, 2)
    cv2.circle(tile, (cx, base), 6, (255, 255, 255), -1)
    heading = values.get("nav_heading")
    if heading is not None:
        rad = np.deg2rad(float(heading))
        cv2.arrowedLine(tile, (cx, base), (int(cx + np.sin(rad) * 60), int(base - np.cos(rad) * 60)),
                        GREEN, 3, tipLength=0.3)
    return tile


def _camera_line(values: dict) -> tuple[str, tuple]:
    """'L 30 USB3 / R 0 USB2! / W 10 USB2!' in amber while any camera is on a
    USB 2 link (a plug not pushed home, or a USB 2 cable)."""
    parts, degraded = [], False
    for tag, key in (("L", "left"), ("R", "right"), ("W", "wide")):
        fps = values.get(f"fps_{key}", 0.0)
        link = int(values.get(f"link_{key}", 0) or 0)
        if link >= 5000:
            usb = "USB3"
        elif link > 0:
            usb, degraded = "USB2!", True
        else:
            usb = "?"
        parts.append(f"{tag} {fps:.0f} {usb}")
    return " / ".join(parts) + (" fps" if not degraded else " fps  reseat!"), (AMBER if degraded else FG)


def _status_tile(w: int, h: int, values: dict) -> np.ndarray:
    tile = np.full((h, w, 3), PANEL, dtype=np.uint8)
    level = str(values.get("safety_level", "UNKNOWN"))
    color = {"CLEAR": GREEN, "WARN": AMBER, "DANGER": RED}.get(level, DIM)
    cv2.rectangle(tile, (10, 34), (w - 10, 84), color, -1)
    cv2.putText(tile, f"SAFETY {level}", (20, 70), FONT, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    rows = [
        ("calibration", values.get("calibration", "absent")),
        ("stereo depth", f"{values.get('depth_ms', 0):.1f} ms  {values.get('fast_hz', 0):.1f} Hz  "
                         f"{values.get('depth_backend', '?')}"),
        ("stereo skew", f"{values.get('skew_ms', 0):.1f} ms  drops {values.get('skew_drops', 0)}"),
        ("cameras", _camera_line(values)),
        ("nearest", f"{values.get('nearest_m', float('inf')):.2f} m {values.get('nearest_dir', '')}"),
        ("approach", values.get("approach", "none")),
        ("slam", values.get("slam", "off")),
        ("navigation", values.get("nav_status", "off (needs calibration)")),
        ("gemma", values.get("gemma", "idle")),
        ("grounding", values.get("grounding", "idle")),
        ("last query", values.get("query_latency", "")),
        ("gpu", f"GR3D {values.get('gr3d_percent', 0)}%  {values.get('gpu_temp_c', 0):.0f} C"),
        ("memory", f"RAM {values.get('ram_used_mb', 0)}/{values.get('ram_total_mb', 0)} MB  "
                   f"swap {values.get('swap_used_mb', 0)} MB"),
    ]
    y = 112
    for label, value in rows:
        text, color = value if isinstance(value, tuple) else (value, FG)
        cv2.putText(tile, label, (14, y), FONT, 0.5, DIM, 1, cv2.LINE_AA)
        cv2.putText(tile, str(text), (150, y), FONT, 0.52, color, 1, cv2.LINE_AA)
        y += 26
    return tile


class Dashboard(threading.Thread):
    """Render loop. `on_question` receives typed questions (called off-thread)."""

    def __init__(self, on_question: Callable[[str], None] | None = None,
                 width: int = 1920, height: int = 1080, fullscreen: bool = True,
                 fps: float = 15.0, window: str = "ARGUS"):
        super().__init__(name="argus-dashboard", daemon=True)
        self.on_question = on_question
        self.width, self.height = width, height
        self.fullscreen = fullscreen
        self.period = 1.0 / fps
        self.window = window
        self.typed = ""
        self.closed = threading.Event()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------ render
    def render(self, snap: Snapshot) -> np.ndarray:
        W, H = self.width, self.height
        gap = 6
        tile_w = (W - 4 * gap) // 3
        row_h = int(H * 0.36)
        log_h = H - 2 * row_h - 4 * gap
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        f, v = snap.frames, snap.values

        left = _fit(f.get("left"), tile_w, row_h)
        _title(left, f"LEFT AR0234  {v.get('fps_left', 0):.0f} fps")
        right = _fit(f.get("right"), tile_w, row_h)
        _title(right, f"RIGHT AR0234  {v.get('fps_right', 0):.0f} fps")
        # Live wide feed normally; the privacy-gated query frame with its
        # detections for a few seconds after each question.
        gated_age = time.monotonic() - float(v.get("gated_at", 0.0))
        show_gated = gated_age < 8.0 and f.get("wide_gated") is not None
        wide_src = f.get("wide_gated") if show_gated else f.get("wide")
        if show_gated and wide_src is not None and snap.detections:
            wide_src = _draw_detections(wide_src, snap.detections)
        wide = _fit(wide_src, tile_w, row_h)
        _title(wide, f"WIDE IMX477  {v.get('fps_wide', 0):.0f} fps   "
                     f"{'privacy-gated query frame' if show_gated else 'live'}",
               MAGENTA if show_gated else FG)

        depth_src = f.get("depth_color")
        depth = _fit(depth_src, tile_w, row_h)
        _title(depth, "DEPTH  " + ("metric" if v.get("calibrated") else "uncalibrated disparity"))
        topdown = _draw_topdown(tile_w, row_h, v)
        _title(topdown, "TOP-DOWN  corridor + SLAM trail")
        status = _status_tile(tile_w, row_h, v)
        _title(status, "STATUS")

        for i, tile in enumerate((left, right, wide)):
            x = gap + i * (tile_w + gap)
            canvas[gap:gap + row_h, x:x + tile_w] = tile
        for i, tile in enumerate((depth, topdown, status)):
            x = gap + i * (tile_w + gap)
            y = 2 * gap + row_h
            canvas[y:y + row_h, x:x + tile_w] = tile

        # Conversation log with typed-question line.
        y0 = 3 * gap + 2 * row_h
        cv2.rectangle(canvas, (gap, y0), (W - gap, H - gap), PANEL, -1)
        lines: list[tuple[str, str]] = []
        for ev in snap.events:
            label = KIND_LABEL.get(ev.kind, ev.kind)
            for j, chunk in enumerate(textwrap.wrap(ev.text, 150) or [""]):
                lines.append((ev.kind, f"{label + ':' if j == 0 else '':8s} {chunk}"))
        max_lines = max(1, (log_h - 44) // 24)
        y = y0 + 24
        for kind, text in lines[-max_lines:]:
            cv2.putText(canvas, text, (gap + 12, y), FONT, 0.58, KIND_COLOR.get(kind, FG), 1, cv2.LINE_AA)
            y += 24
        prompt = f"> {self.typed}_" if self.on_question else "(microphone mode)"
        cv2.rectangle(canvas, (gap, H - gap - 34), (W - gap, H - gap), (0, 0, 0), -1)
        cv2.putText(canvas, prompt, (gap + 12, H - gap - 10), FONT, 0.62, CYAN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "type a question + Enter  |  q quits", (W - 420, H - gap - 10),
                    FONT, 0.5, DIM, 1, cv2.LINE_AA)
        return canvas

    # ------------------------------------------------------------ loop
    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window, self.width, self.height)
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                cv2.imshow(self.window, self.render(bus.snapshot()))
                key = cv2.waitKey(max(1, int(self.period * 1000 * 0.5))) & 0xFF
                if key != 255:
                    if not self._handle_key(key):
                        break
                try:
                    if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                dt = time.perf_counter() - t0
                if dt < self.period:
                    time.sleep(self.period - dt)
        finally:
            cv2.destroyAllWindows()
            self.closed.set()

    def _handle_key(self, key: int) -> bool:
        if key in (13, 10):
            question = self.typed.strip()
            self.typed = ""
            if question and self.on_question:
                threading.Thread(target=self.on_question, args=(question,),
                                 name="dashboard-question", daemon=True).start()
            return True
        if key == 27:
            if not self.typed:
                return False
            self.typed = ""
            return True
        if key in (8, 127):
            self.typed = self.typed[:-1]
            return True
        if key == ord("q") and not self.typed:
            return False
        if 32 <= key < 127 and self.on_question:
            self.typed += chr(key)
        return True

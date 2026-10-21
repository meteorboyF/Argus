"""First-person demo view (M8): `python3 -m argus run --demo [--record out.mp4]`.

For the Wizard-of-Oz comparison video and the evaluation sessions:

    +--------------------------------------------+--------------+
    |                                            |  DEPTH inset |
    |   WIDE camera POV (faces blurred)          +--------------+
    |   [ HAZARD BANNER  <-  Stop. Left.  ((o)) ] |  CORRIDOR    |
    |                                            +--------------+
    |                                            |  latency     |
    +--------------------------------------------+  chips       |
    |  YOU:   find me a seat                                    |
    |  ARGUS: Seat, slightly right, 4 steps.   (lit while spoken)|
    +-----------------------------------------------------------+

Face blur here is for DISPLAY and recording only (people near the wearer
appear in the demo video). It runs InsightFace on a background thread at a
few Hz and applies the latest boxes, padded, to every frame.

--record writes the composed frames to a video and every TTS sentence and
earcon (captured at playback time) to an audio timeline, then muxes them
with ffmpeg into one MP4.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np

from .dashboard import FONT, Dashboard, _draw_topdown, _fit
from .telemetry import Snapshot, bus

W, H = 1920, 1080
POV_W, POV_H = 1440, 810
SIDE_W = W - POV_W
CAP_H = H - POV_H

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
YELLOW, RED, CYAN = (0, 230, 255), (40, 40, 235), (230, 200, 60)


# ---------------------------------------------------------------------------
# Display face blur
# ---------------------------------------------------------------------------
class FaceBlurCache(threading.Thread):
    """Latest face boxes on the wide frame, refreshed at ~3 Hz off-thread."""

    def __init__(self, privacy_cfg, period_s: float = 0.33):
        super().__init__(daemon=True, name="demo-faceblur")
        self.period_s = period_s
        self.boxes: list[tuple[int, int, int, int]] = []
        self._frame = None
        self._lock = threading.Lock()
        self._halt = threading.Event()   # not _stop: that shadows Thread._stop() and breaks join()
        self.ok = False
        try:
            from .privacy import PrivacyGate
            self._gate = PrivacyGate(privacy_cfg)
            self.ok = self._gate.ready
        except Exception:  # noqa: BLE001
            self._gate = None

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def run(self):
        while not self._halt.is_set() and self.ok:
            with self._lock:
                f, self._frame = self._frame, None
            if f is None:
                time.sleep(0.03)
                continue
            try:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                faces = self._gate._face.get(rgb)
                self.boxes = [tuple(int(v) for v in x.bbox) for x in faces]
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.period_s)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.ok:
            # No detector: blur the whole POV rather than show faces.
            return cv2.GaussianBlur(frame, (0, 0), 12)
        out = frame
        for x1, y1, x2, y2 in self.boxes:
            pw, ph = int((x2 - x1) * 0.35), int((y2 - y1) * 0.35)   # pad: faces move
            x1, y1 = max(0, x1 - pw), max(0, y1 - 2 * ph)            # more above: hair/forehead
            x2, y2 = min(frame.shape[1], x2 + pw), min(frame.shape[0], y2 + ph)
            if x2 > x1 and y2 > y1:
                if out is frame:
                    out = frame.copy()
                out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (0, 0), 18)
        return out

    def stop(self):
        self._halt.set()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _text(img, s, org, scale, color, thick=2, bg=None, pad=10):
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    if bg is not None:
        cv2.rectangle(img, (org[0] - pad, org[1] - th - pad),
                      (org[0] + tw + pad, org[1] + base + pad // 2), bg, -1)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)
    return tw


def _wrap(s: str, scale: float, max_w: int, thick: int = 2) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cv2.getTextSize(trial, FONT, scale, thick)[0][0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _arrow(img, side: str, center, size, color):
    cx, cy = center
    if side == "left":
        pts = [(cx - size, cy), (cx + size // 2, cy - size), (cx + size // 2, cy + size)]
    elif side == "right":
        pts = [(cx + size, cy), (cx - size // 2, cy - size), (cx - size // 2, cy + size)]
    else:
        pts = [(cx, cy - size), (cx - size, cy + size // 2), (cx + size, cy + size // 2)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def _earcon_icon(img, center, r, color):
    cv2.circle(img, center, r // 4, color, -1, cv2.LINE_AA)
    for k in (2, 3):
        cv2.ellipse(img, center, (r * k // 3, r * k // 3), 0, -45, 45, color, 3, cv2.LINE_AA)
        cv2.ellipse(img, center, (r * k // 3, r * k // 3), 0, 135, 225, color, 3, cv2.LINE_AA)


def hazard_side(text: str) -> str:
    t = text.lower()
    return "left" if "left" in t else "right" if "right" in t else "ahead"


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------
class DemoView(Dashboard):
    def __init__(self, on_question=None, privacy_cfg=None, fullscreen: bool = True,
                 record_path: str | None = None, fps: float = 15.0):
        super().__init__(on_question=on_question, width=W, height=H, fullscreen=fullscreen,
                         fps=fps, window="ARGUS demo")
        # A stray Esc/q while switching windows ended two voice tests on
        # 2026-10-21. In the demo nobody at the keyboard should be able to end
        # the wearer's session by accident: stop it with Ctrl+C instead.
        self.allow_key_quit = False
        self.blur = FaceBlurCache(privacy_cfg) if privacy_cfg is not None else None
        if self.blur is not None:
            self.blur.start()
        self.recorder = Recorder(record_path, fps) if record_path else None

    # ------------------------------------------------------------ render
    def render(self, snap: Snapshot) -> np.ndarray:
        v, f = snap.values, snap.frames
        now = time.monotonic()
        canvas = np.zeros((H, W, 3), np.uint8)

        # POV
        wide = f.get("wide")
        if wide is not None and self.blur is not None:
            self.blur.submit(wide)
            wide = self.blur.apply(wide)
        canvas[:POV_H, :POV_W] = _fit(wide, POV_W, POV_H)
        _text(canvas, "ARGUS  first-person view", (24, 44), 0.9, WHITE, 2, bg=BLACK)

        # Hazard banner (2.5 s after the last safety line).
        safety = [e for e in snap.events if e.kind == "safety"]
        if safety and now - safety[-1].when < 2.5:
            ev = safety[-1]
            flash = int((now - ev.when) * 4) % 2 == 0
            col = RED if flash else (20, 20, 160)
            y0, y1 = POV_H - 190, POV_H - 40
            cv2.rectangle(canvas, (60, y0), (POV_W - 60, y1), col, -1)
            cv2.rectangle(canvas, (60, y0), (POV_W - 60, y1), WHITE, 4)
            _arrow(canvas, hazard_side(ev.text), (150, (y0 + y1) // 2), 45, WHITE)
            _text(canvas, ev.text, (240, (y0 + y1) // 2 + 22), 2.0, WHITE, 5)
            _earcon_icon(canvas, (POV_W - 140, (y0 + y1) // 2), 60, WHITE)

        # Side column: depth, corridor, chips.
        inset_h = 250
        depth = _fit(f.get("depth_color"), SIDE_W, inset_h)
        _text(depth, "DEPTH" + ("" if v.get("calibrated") else " (uncalibrated)"), (12, 30),
              0.65, WHITE, 2, bg=BLACK, pad=6)
        canvas[:inset_h, POV_W:] = depth
        topdown = _draw_topdown(SIDE_W, inset_h, v)
        _text(topdown, "CORRIDOR", (12, 30), 0.65, WHITE, 2, bg=BLACK, pad=6)
        canvas[inset_h:2 * inset_h, POV_W:] = topdown
        y = 2 * inset_h + 40
        chips = [
            ("answer", v.get("query_latency")),
            ("listen/STT", v.get("stt_latency")),
            ("depth", f"{v['depth_ms']:.0f} ms" if v.get("depth_ms") else None),
            ("safety loop", f"{v['fast_hz']:.0f} Hz" if v.get("fast_hz") else None),
            ("reading", v.get("reading")),
            ("seat", v.get("seat")),
            ("people", v.get("movers")),
        ]
        for name, val in chips:
            if not val or y > POV_H - 20:
                continue
            s = f"{name}: {val}"
            s = s if len(s) <= 40 else s[:37] + "..."     # Hershey fonts are ASCII-only
            _text(canvas, s, (POV_W + 16, y), 0.58, WHITE, 1, bg=(50, 50, 50), pad=6)
            y += 38

        # Captions.
        cv2.rectangle(canvas, (0, POV_H), (W, H), (12, 12, 12), -1)
        user = next((e.text for e in reversed(snap.events) if e.kind == "user"), "")
        speaking = v.get("speaking") or ""
        last_argus = next((e.text for e in reversed(snap.events)
                           if e.kind in ("argus", "safety")), "")
        argus_text = speaking or last_argus
        lit = bool(speaking)
        yy = POV_H + 70
        _text(canvas, "YOU:", (40, yy), 1.3, CYAN, 3)
        for line in _wrap(user or "...", 1.3, W - 260)[:1]:
            _text(canvas, line, (200, yy), 1.3, WHITE, 3)
        yy += 90
        _text(canvas, "ARGUS:", (40, yy), 1.3, YELLOW, 3)
        for line in _wrap(argus_text or "...", 1.3, W - 260)[:2]:
            if lit:
                (tw, th), _ = cv2.getTextSize(line, FONT, 1.3, 3)
                cv2.rectangle(canvas, (190, yy - th - 12), (210 + tw, yy + 14), (0, 90, 120), -1)
            _text(canvas, line, (200, yy), 1.3, WHITE if lit else (180, 180, 180), 3)
            yy += 62
        if self.on_question is not None and self.typed:
            _text(canvas, f"> {self.typed}_", (40, H - 20), 0.8, CYAN, 2)
        if self.recorder is not None:
            cv2.circle(canvas, (W - 40, 40), 14, (0, 0, 255), -1)
            self.recorder.write(canvas)
        return canvas

    def stop(self):
        super().stop()
        if self.blur is not None:
            self.blur.stop()

    def finish(self) -> str | None:
        return self.recorder.finish() if self.recorder is not None else None


# ---------------------------------------------------------------------------
# Recording (video + spoken audio)
# ---------------------------------------------------------------------------
class Recorder:
    """Constant-rate video from rendered frames + an audio timeline built
    from the exact PCM that was played (TTS sentences and earcons).

    The Orin Nano has no hardware video encoder, and 1080p MJPG cost 138 ms
    per frame in the render thread. Frames are therefore downscaled to
    960x540 and piped from a background thread to an ffmpeg x264
    'ultrafast' process; the render loop only enqueues (and frames are
    dropped, never waited for, if the encoder falls behind).
    """

    SR = 22050
    SIZE = (960, 540)

    def __init__(self, out_path: str, fps: float):
        import queue
        self.out = Path(out_path)
        self.fps = fps
        self.tmp = Path(tempfile.mkdtemp(prefix="argus_rec_"))
        self.video_path = self.tmp / "video.mp4"
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{self.SIZE[0]}x{self.SIZE[1]}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
             str(self.video_path)], stdin=subprocess.PIPE)
        self.t0 = time.monotonic()
        self.frames = 0
        self.dropped = 0
        self.audio: list[tuple[float, np.ndarray, int]] = []
        self._lock = threading.Lock()
        self._q: "queue.Queue" = queue.Queue(maxsize=30)
        self._last = None
        self._thread = threading.Thread(target=self._encode, daemon=True, name="demo-rec")
        self._thread.start()
        AUDIO_TAPS.append(self.on_audio)

    def write(self, frame: np.ndarray) -> None:
        small = cv2.resize(frame, self.SIZE, interpolation=cv2.INTER_AREA)
        try:
            self._q.put_nowait((time.monotonic(), small))
        except Exception:  # noqa: BLE001 — queue full: drop, never block the view
            self.dropped += 1

    def _encode(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            t, frame = item
            # Keep wall-clock sync: repeat frames to fill any gap.
            target = int((t - self.t0) * self.fps) + 1
            while self.frames < target:
                try:
                    self.proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, ValueError):
                    return
                self.frames += 1

    def on_audio(self, pcm: np.ndarray, sr: int, started: float) -> None:
        with self._lock:
            self.audio.append((started - self.t0, np.asarray(pcm, np.float32), sr))

    def finish(self) -> str:
        if self.on_audio in AUDIO_TAPS:
            AUDIO_TAPS.remove(self.on_audio)
        self._q.put(None)
        self._thread.join(timeout=30)
        self.proc.stdin.close()
        self.proc.wait(timeout=60)
        dur = self.frames / self.fps
        track = np.zeros(int(dur * self.SR) + self.SR, np.float32)
        for start, pcm, sr in self.audio:
            if sr != self.SR:
                n = int(len(pcm) * self.SR / sr)
                pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm)
            i = int(max(0.0, start) * self.SR)
            j = min(len(track), i + len(pcm))
            track[i:j] += pcm[:j - i]
        track = np.clip(track, -1, 1)
        wav = self.tmp / "audio.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SR)
            w.writeframes((track * 32767).astype(np.int16).tobytes())
        self.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.video_path),
                        "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest",
                        str(self.out)], check=True)
        return str(self.out)


# Audio taps: speech/earcon playback calls these with (pcm, sample_rate,
# monotonic start time). Empty unless a recording is running.
AUDIO_TAPS: list = []


def tap_audio(pcm: np.ndarray, sr: int) -> None:
    started = time.monotonic()
    for tap in list(AUDIO_TAPS):
        try:
            tap(pcm, sr, started)
        except Exception:  # noqa: BLE001
            pass

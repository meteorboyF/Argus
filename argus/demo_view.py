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

--record writes the composed frames, continuous default microphone input,
and ARGUS playback audio into one MP4.
"""
from __future__ import annotations

import math
import re
import signal
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


def reading_preview(frame, snap: Snapshot, source: str) -> np.ndarray:
    """Local diagnostic only; never passed to Recorder or the image model.

    Live camera above the last source capture; right panel shows the actual
    voice-triggered OCR output. No additional OCR work runs in the UI thread.
    """
    from .sensitive import mask_text
    canvas = np.zeros((900, 1440, 3), np.uint8)
    canvas[40:410, :720] = _fit(frame, 720, 370)
    _text(canvas, f"LIVE OCR CAMERA: {source.upper()}" if frame is not None else
          "OCR CAMERA: waiting for fresh frame", (12, 30), .65, CYAN, 2)
    capture_at = float(snap.values.get("reading_capture_at", 0))
    capture = snap.frames.get("reading_capture") if capture_at else None
    canvas[460:830, :720] = _fit(capture, 720, 370)
    age = max(0, time.monotonic()-capture_at)
    label = f"LAST CAPTURE: {age:.0f}s ago" if capture_at else "LAST CAPTURE: none yet"
    _text(canvas, label, (12, 450), .65, CYAN, 2)
    _text(canvas, "ACTUAL OCR TEXT (last completed recognition)", (740, 30), .6, YELLOW, 1)
    status = mask_text(str(snap.values.get("reading_ocr_status", "Say: Argus, read this")))
    y = 65
    for line in _wrap(status, .6, 670, 1):
        _text(canvas, line, (740, y), .6, CYAN, 1)
        y += 26
    y += 20
    text = mask_text(str(snap.values.get("reading_ocr_text", "")))
    lines = [part for line in text.splitlines() for part in _wrap(line, .65, 670, 1)]
    limit = max(0, (825-y)//30)
    for line in lines[:limit]:
        _text(canvas, line, (740, y), .65, WHITE, 1)
        y += 30
    if len(lines) > limit:
        _text(canvas, "[more text below; page playback continues]", (740, 850), .55, YELLOW, 1)
    _text(canvas, "Live view + last captured image + recognized text. Say 'Argus, read this' to update.",
          (12, 880), .6, WHITE, 1)
    return canvas


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------
class DemoView(Dashboard):
    def __init__(self, on_question=None, privacy_cfg=None, fullscreen: bool = True,
                 record_path: str | None = None, fps: float = 15.0, reading_rig=None,
                 pov_source: str = "wide", feature: str = "auto"):
        from .recording_layout import RecordingLayout
        self.layout = RecordingLayout(feature)
        self.show_diagnostics = False
        if pov_source not in ("wide","left"):
            raise ValueError("demo source must be wide or left")
        self.pov_source = pov_source
        super().__init__(on_question=on_question, width=W, height=H, fullscreen=fullscreen,
                         fps=fps, window="ARGUS demo")
        # A stray Esc/q while switching windows ended two voice tests on
        # 2026-10-21. In the demo nobody at the keyboard should be able to end
        # the wearer's session by accident: stop it with Ctrl+C instead.
        self.allow_key_quit = False
        self.reading_rig = reading_rig
        self._reading_window = "ARGUS OCR - live camera and recognition"
        self._reading_window_open = False
        self.blur = FaceBlurCache(privacy_cfg) if privacy_cfg is not None else None
        if self.blur is not None:
            self.blur.start()
        self.recorder = Recorder(record_path, fps) if record_path else None
        self._recording_finalizers = []
        self._reset_at = -10.0

    # ------------------------------------------------------------ render
    def render(self, snap: Snapshot) -> np.ndarray:
        frame = snap.frames.get(self.pov_source)
        if frame is not None and self.blur is not None:
            self.blur.submit(frame)
            frame = self.blur.apply(frame)
        canvas = self.layout.render(
            snap, frame, source=self.pov_source, recording=self.recorder is not None,
            elapsed=time.monotonic()-self.recorder.t0 if self.recorder else 0,
            diagnostics=self.show_diagnostics,
            privacy_ready=bool(self.blur and self.blur.ok),
            face_count=len(self.blur.boxes) if self.blur and self.blur.ok else None)
        if self.on_question is not None and self.typed:
            cv2.rectangle(canvas,(36,H-62),(W-36,H-12),(20,30,32),-1)
            _text(canvas, f"> {self.typed}_", (48,H-28),.8,CYAN,2)
        if snap.values.get("study_instruction"):
            from .guide import draw_banner
            draw_banner(canvas,snap.values["study_instruction"],
                        progress=snap.values.get("study_progress"))
        if self.recorder is not None:
            self.recorder.write(canvas)
        if self.reading_rig is not None:
            if not self._reading_window_open:
                cv2.namedWindow(self._reading_window,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._reading_window,1440,900)
                self._reading_window_open=True
            cv2.imshow(self._reading_window,reading_preview(
                self.reading_rig.get_reading_frame(),snap,
                self.reading_rig.cfg.reading_source))
        return canvas

    def _handle_key(self,key):
        if key in (ord('r'), ord('R')) and self.recorder is not None:
            self._reset_recording()
            return True
        if key == 9:
            self.show_diagnostics=not self.show_diagnostics
            return True
        return super()._handle_key(key)

    def _reset_recording(self):
        now = time.monotonic()
        if now - self._reset_at < 1.0 or any(t.is_alive() for t in self._recording_finalizers):
            return
        old = self.recorder
        match = re.match(r'^(.*-take-)(\d+)$', old.out.stem)
        prefix = match[1] if match else old.out.stem + '-take-'
        number = int(match[2]) + 1 if match else 2
        while True:
            path = old.out.with_name(f'{prefix}{number:02d}{old.out.suffix}')
            if not path.exists() and not path.with_suffix('.jsonl').exists():
                break
            number += 1
        try:
            new = Recorder(str(path), old.fps)
        except Exception as exc:
            print(f'[demo] recording reset failed: {exc}')
            return
        self.recorder = new
        self.typed = ''
        bus.clear_interaction_display()
        self._reset_at = now
        # Save the previous clip away from rendering and microphone/STT work.
        def finalize():
            try:
                print(f'[demo] recording saved: {old.finish()}')
            except Exception as exc:
                print(f'[demo] previous recording failed: {exc}')
        worker = threading.Thread(target=finalize, name='demo-save-take', daemon=True)
        self._recording_finalizers = [worker]
        worker.start()
        bus.record('recording_reset', previous=str(old.out), current=str(path))

    def stop(self):
        super().stop()
        if self.blur is not None:
            self.blur.stop()

    def finish(self) -> str | None:
        for worker in self._recording_finalizers:
            worker.join()
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
        self.mic_path = self.tmp / "microphone.wav"
        self._mic_log = open(self.tmp / "microphone.log", "wb")
        self.mic_started = time.monotonic() - self.t0
        # PulseAudio shares the source with wake/STT capture. Write to disk,
        # rather than retaining continuous microphone PCM in Jetson RAM.
        self.mic_proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "pulse", "-i", "default",
             "-ac", "1", "-ar", str(self.SR), "-c:a", "pcm_s16le", str(self.mic_path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self._mic_log)
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
        mic_stopped = time.monotonic() - self.t0
        if self.mic_proc.poll() is None:
            self.mic_proc.send_signal(signal.SIGINT)
        self.mic_proc.wait(timeout=15)
        self._mic_log.close()
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
        if not self.mic_path.exists() or self.mic_path.stat().st_size <= 44:
            raise RuntimeError("Microphone recording failed; raw video and ARGUS audio "
                               f"are preserved in {self.tmp}. See microphone.log.")
        with wave.open(str(self.mic_path), "rb") as mic:
            mic_duration = mic.getnframes() / mic.getframerate()
        # FFmpeg/Pulse startup is not instantaneous. Anchor the captured
        # samples to stop time instead of placing them early on the timeline.
        delay = int(max(self.mic_started, mic_stopped - mic_duration) * 1000)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.video_path),
                        "-i", str(wav), "-i", str(self.mic_path),
                        "-filter_complex", f"[2:a]adelay={delay}:all=1[mic];"
                        "[1:a][mic]amix=inputs=2:duration=first:normalize=0,"
                        "alimiter=limit=0.95:level=false[mix]",
                        "-map", "0:v:0", "-map", "[mix]",
                        "-c:v", "copy", "-c:a", "aac", "-shortest",
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

"""Reading mode (M5): frame the page by voice, capture a still, read it aloud.

Flow ("read this"):
  1. Coach: text detection on the live wide stream at a few Hz. Spoken cues
     ("move the page to the right", "bring it closer", "hold still") until
     the text block is inside the frame, large enough, sharp and still.
  2. Capture: earcon, then a 12 MP still (CameraRig.capture_still).
  3. OCR (argus.ocr) on the still; text spoken sentence by sentence.
     "stop", "pause", "continue", "repeat" control playback.
  4. Questions about the page ("what's the expiry date?") go to Gemma as
     *text only*, with sensitive numbers masked, never as an image.

Directions are from the wearer's point of view: the wide camera faces the
same way as the wearer, so image-right is the wearer's right.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import earcons
from .ocr import OCREngine, Reading, speakable_chunks
from .sensitive import mask_text
from .telemetry import bus

_READ_RE = re.compile(
    r"^\s*(?:please\s+)?(?:read(?:\s+(?:this|it|that|the\s+\w+|for\s+me))*|"
    r"what\s+does\s+(?:it|this|that|the\s+\w+)\s+say)\b", re.IGNORECASE)
_CONTROL_RE = re.compile(r"^\s*(stop|pause|continue|resume|repeat|again)\b", re.IGNORECASE)
_PAGE_Q_RE = re.compile(
    r"\b(page|label|sign|text|written|says?|expir\w*|date|price|dose|dosage|amount|total|"
    r"name|address|ingredients?|instructions?)\b", re.IGNORECASE)


def is_read_request(q: str) -> bool:
    return bool(_READ_RE.search(q))


def control_word(q: str) -> str | None:
    m = _CONTROL_RE.search(q)
    if not m:
        return None
    w = m.group(1).lower()
    return {"resume": "continue", "again": "repeat"}.get(w, w)


def is_page_question(q: str) -> bool:
    return bool(_PAGE_Q_RE.search(q)) and not is_read_request(q)


# ---------------------------------------------------------------------------
# Framing advice (pure, testable)
# ---------------------------------------------------------------------------
@dataclass
class FrameCheck:
    boxes: list            # text boxes (4x2) in the analysed frame
    width: int
    height: int
    sharpness: float       # Laplacian variance inside the text block
    motion: float          # mean abs grey difference to the previous frame


@dataclass
class FrameLimits:
    edge_frac: float = 0.03      # text this close to an edge is probably cut off
    # The 12 MP still has ~2x the linear resolution of the 1080p live view
    # the coach analyses, so a block 20% of the live width is readable.
    min_width_frac: float = 0.20 # text block narrower than this: too far away
    # The B0459 has a fixed-focus lens (no V4L2 focus control): when the
    # page is still but blurry and already large, it is too close to focus.
    close_width_frac: float = 0.45
    min_sharpness: float = 60.0
    max_motion: float = 6.0


def framing_advice(fc: FrameCheck, lim: FrameLimits = FrameLimits()) -> tuple[str, bool]:
    """(spoken cue, ready_to_capture). One cue at a time, most important first."""
    if not fc.boxes:
        return ("I can't see any text yet. Hold the page in front of you, "
                "about thirty centimetres away."), False
    pts = np.concatenate([np.asarray(b, np.float32).reshape(-1, 2) for b in fc.boxes])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    ex, ey = lim.edge_frac * fc.width, lim.edge_frac * fc.height
    cut_left, cut_right = x0 < ex, x1 > fc.width - ex
    cut_top, cut_bottom = y0 < ey, y1 > fc.height - ey
    if fc.motion > lim.max_motion:
        return "Hold still.", False
    if (cut_left and cut_right) or (cut_top and cut_bottom):
        return "Move the page a little farther away.", False
    if cut_left:
        return "Move the page to your right.", False
    if cut_right:
        return "Move the page to your left.", False
    if cut_top:
        return "Move the page down, or lift your head a little.", False
    if cut_bottom:
        return "Move the page up, or lower your head a little.", False
    if (x1 - x0) < lim.min_width_frac * fc.width:
        return "Bring the page closer.", False
    if fc.sharpness < lim.min_sharpness:
        if (x1 - x0) > lim.close_width_frac * fc.width:
            return "Move the page a little farther away. It is too close to focus.", False
        return "Hold still, the picture is blurry.", False
    return "Hold still. Capturing.", True


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class ReadingSession:
    """Owns one 'read this' interaction and the playback that follows."""

    def __init__(self, rig, speaker, ocr: OCREngine | None = None,
                 coach_timeout_s: float = 30.0):
        self.rig = rig
        self.speaker = speaker
        self.ocr = ocr or OCREngine()
        self.coach_timeout_s = coach_timeout_s
        self.last: Reading | None = None
        self.last_at = 0.0
        self._chunks: list[str] = []
        self._pos = 0
        self._playing = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------- coach
    def _detect(self, frame: np.ndarray) -> list:
        res, _ = self.ocr._ocr(frame, use_det=True, use_cls=False, use_rec=False)
        return [np.asarray(b, np.float32) for b in (res or [])]

    def coach(self) -> bool:
        """Speak framing cues until the text is framed; False on timeout."""
        prev = None
        said = ""
        said_at = 0.0
        t0 = time.monotonic()
        steady = 0
        while time.monotonic() - t0 < self.coach_timeout_s:
            frame = self.rig.get_wide_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            small = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            motion = float(np.abs(gray.astype(np.int16) - prev).mean()) if prev is not None else 0.0
            prev = gray.astype(np.int16)
            boxes = self._detect(small)
            sharp = 0.0
            if boxes:
                pts = np.concatenate(boxes).astype(int)
                x, y, w, h = cv2.boundingRect(pts)
                roi = gray[max(0, y):y + h, max(0, x):x + w]
                if roi.size:
                    sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            cue, ready = framing_advice(FrameCheck(boxes, small.shape[1], small.shape[0],
                                                   sharp, motion))
            bus.set(reading=cue)
            steady = steady + 1 if ready else 0
            if steady >= 2:            # two good checks in a row
                return True
            now = time.monotonic()
            if cue != said and (now - said_at > 2.0) and not self.speaker.busy:
                self.speaker.speak(cue)
                said, said_at = cue, now
        return False

    # -------------------------------------------------------------- read
    def read(self, coach: bool = True, intro: str | None = "Reading mode. Hold the page "
             "in front of you, about arm's length away.",
             on_finished=None) -> Reading | None:
        """Coach, capture, OCR, then speak. `on_finished` runs once the whole
        page has been spoken (not when it was stopped)."""
        self.stop()
        self._on_finished = on_finished
        if intro:
            self.speaker.speak(intro)
        if coach and not self.coach():
            self.speaker.speak("I couldn't frame the text. Try again with more light.")
            return None
        earcons.play("capturing")
        t0 = time.perf_counter()
        still = self.rig.capture_still()
        if still is None:
            self.speaker.speak("The camera did not give me a picture.")
            return None
        capture_s = time.perf_counter() - t0
        reading = self.ocr.read(still)
        bus.record("reading", capture_s=capture_s, ocr_s=reading.elapsed_s,
                   lines=len(reading.lines), sensitive=reading.sensitive)
        bus.set(reading=f"{len(reading.lines)} lines in {time.perf_counter() - t0:.1f}s "
                        "(capture + OCR)")
        # Only the masked text ever reaches logs or the screen.
        bus.log("argus", "[page] " + reading.masked_text.replace("\n", " / ")[:300])
        if not reading.lines:
            self.speaker.speak("I found no readable text.")
            return None
        self.last, self.last_at = reading, time.monotonic()
        self._chunks = speakable_chunks(reading.lines)
        self._pos = 0
        self._start_playback()
        return reading

    # -------------------------------------------------------------- playback
    def _start_playback(self):
        self._stop.clear()
        self._playing.set()
        self._thread = threading.Thread(target=self._play_loop, daemon=True, name="reading")
        self._thread.start()

    def _play_loop(self):
        while self._pos < len(self._chunks) and not self._stop.is_set():
            if not self._playing.wait(0.2):
                continue
            self.speaker.speak(self._chunks[self._pos])
            self.speaker.wait_until_idle(timeout=30.0)
            if self._playing.is_set() and not self._stop.is_set():
                self._pos += 1
        if self._pos >= len(self._chunks) and not self._stop.is_set():
            earcons.play("done")
            cb = getattr(self, "_on_finished", None)
            if cb is not None:
                try:
                    cb()
                except Exception as e:  # noqa: BLE001
                    print(f"[reading] on_finished failed: {e}")

    def control(self, word: str) -> None:
        if word == "stop":
            self.stop()
        elif word == "pause":
            self._playing.clear()
            self.speaker.cancel_normal()
        elif word == "continue" and self._chunks:
            if self._thread is None or not self._thread.is_alive():
                self._start_playback()
            self._playing.set()
        elif word == "repeat" and self._chunks:
            self.stop()
            self._pos = 0
            self._start_playback()

    def stop(self) -> None:
        self._stop.set()
        self._playing.set()
        self.speaker.cancel_normal()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def active(self) -> bool:
        return self.last is not None and time.monotonic() - self.last_at < 300.0

    def page_context(self) -> str:
        """Masked page text for a text-only question to Gemma."""
        return mask_text(self.last.text) if self.last else ""

"""Guided human interaction for physical setup steps.

Every tool a person interacts with (calibration, alignment, rig doctor, skew
test) goes through `Guide`, so that whoever is holding a board or wearing the
rig is never left in front of a silent window:

- instructions are spoken through Piper, and only when they change (or after
  `repeat_s` if the person has not acted on them),
- the same instruction is drawn in large text on the monitor,
- progress feedback is continuous ("captured 4 of 15"),
- every session ends with a spoken DONE or FAILED plus the reason.

The person wearing the rig cannot read a monitor; the person at the keyboard
may not hear the bone-conduction headset. Both channels carry the same words.
"""
from __future__ import annotations

import os
import textwrap
import time

import cv2
import numpy as np

from .config import SpeechConfig


def display_available() -> bool:
    return bool(os.environ.get("DISPLAY"))


def wrap_lines(text: str, width: int) -> list[str]:
    out: list[str] = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, width=width) or [""])
    return out


def draw_banner(canvas: np.ndarray, headline: str, sub: str = "",
                progress: float | None = None, colour=(255, 255, 255)) -> np.ndarray:
    """Large instruction text across the top of `canvas`, readable from ~2 m."""
    h, w = canvas.shape[:2]
    scale = max(0.8, w / 1100.0)
    chars = max(16, int(w / (24 * scale)))
    head = wrap_lines(headline, chars)[:3]
    subs = wrap_lines(sub, int(chars * 1.6))[:3] if sub else []
    line_h = int(46 * scale)
    sub_h = int(30 * scale)
    box_h = 20 + line_h * len(head) + sub_h * len(subs) + (int(22 * scale) if progress is not None else 0)
    shade = canvas.copy()
    cv2.rectangle(shade, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.8, canvas, 0.2, 0, canvas)
    y = 10
    for line in head:
        y += line_h
        cv2.putText(canvas, line, (20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * scale,
                    colour, max(2, int(3 * scale)), cv2.LINE_AA)
    for line in subs:
        y += sub_h
        cv2.putText(canvas, line, (22, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * scale,
                    (200, 200, 200), max(1, int(2 * scale)), cv2.LINE_AA)
    if progress is not None:
        p = float(np.clip(progress, 0.0, 1.0))
        y0 = y + 6
        cv2.rectangle(canvas, (20, y0), (w - 20, y0 + int(12 * scale)), (90, 90, 90), -1)
        cv2.rectangle(canvas, (20, y0), (20 + int((w - 40) * p), y0 + int(12 * scale)),
                      (80, 220, 80), -1)
    return canvas


class Guide:
    """Speak + show one instruction at a time; rate-limited, change-driven.

    `say()` is cheap to call every frame: it only speaks when the text differs
    from the last spoken instruction (after `min_gap_s`) or when the same
    instruction has gone unheeded for `repeat_s`. While a sentence is still
    playing, a *changed* instruction waits rather than queueing a backlog of
    stale directions.
    """

    def __init__(self, speech_cfg: SpeechConfig | None, title: str = "ARGUS",
                 speak: bool = True, show: bool = True, fullscreen: bool = True,
                 min_gap_s: float = 1.5, repeat_s: float = 8.0, speaker=None):
        self.title = title
        self.min_gap_s = min_gap_s
        self.repeat_s = repeat_s
        self.show_enabled = show and display_available()
        self._spoken = ""
        self._spoken_at = 0.0
        self.headline = ""
        self.sub = ""
        self.progress: float | None = None
        self.speaker = speaker
        if self.speaker is None and speak and speech_cfg is not None:
            from .speech import Speaker
            self.speaker = Speaker(speech_cfg, enabled=True)
            if not self.speaker.enabled:
                print("[guide] Piper unavailable: instructions are on screen only")
        if self.show_enabled:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ------------------------------------------------------------- speech
    @property
    def speaking(self) -> bool:
        return self.speaker is not None and self.speaker.busy

    def say(self, text: str, force: bool = False, wait: bool = False) -> bool:
        """Show `text` as the headline and speak it if it is news. Returns
        True when speech was queued."""
        self.headline = text
        now = time.monotonic()
        changed = text != self._spoken
        due = (force
               or (changed and now - self._spoken_at >= self.min_gap_s and not self.speaking)
               or (not changed and now - self._spoken_at >= self.repeat_s and not self.speaking))
        if not due:
            return False
        if changed or force:
            print(f"[guide] {text}")
        self._spoken, self._spoken_at = text, now
        if self.speaker is not None:
            self.speaker.speak(text)
            if wait:
                self.speaker.wait_until_idle(timeout=30.0)
        return True

    def announce(self, text: str) -> None:
        """Speak and print unconditionally, then wait for playback."""
        self.say(text, force=True, wait=True)

    # ------------------------------------------------------------- screen
    def show(self, frame: np.ndarray | None = None, sub: str | None = None,
             progress: float | None = None, colour=(255, 255, 255),
             delay_ms: int = 1) -> int:
        """Draw the current instruction over `frame`; returns the key code."""
        if sub is not None:
            self.sub = sub
        if progress is not None:
            self.progress = progress
        if not self.show_enabled:
            return -1
        canvas = (np.zeros((1080, 1920, 3), np.uint8) if frame is None else frame.copy())
        draw_banner(canvas, self.headline, self.sub, self.progress, colour)
        cv2.imshow(self.title, canvas)
        return cv2.waitKey(delay_ms) & 0xFF

    # ------------------------------------------------------------- endings
    def done(self, message: str = "") -> None:
        text = f"Done. {message}".strip()
        self.say(text, force=True)
        self.show(colour=(80, 255, 80), delay_ms=1)
        if self.speaker is not None:
            self.speaker.wait_until_idle(timeout=30.0)

    def failed(self, reason: str) -> None:
        text = f"Failed. {reason}".strip()
        self.say(text, force=True)
        self.show(colour=(60, 60, 255), delay_ms=1)
        if self.speaker is not None:
            self.speaker.wait_until_idle(timeout=30.0)

    def close(self) -> None:
        if self.speaker is not None:
            self.speaker.wait_until_idle(timeout=30.0)
            self.speaker.stop()
        if self.show_enabled:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

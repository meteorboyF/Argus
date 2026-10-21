"""Short non-speech audio cues (M6): faster to hear than words.

Generated sine/chirp tones played through PulseAudio (`paplay`, same path as
Piper) on a background thread so callers never block. A missing audio device
must never break the caller: failures are swallowed and printed.
"""
from __future__ import annotations

import shutil
import subprocess
import threading

import numpy as np

SR = 22050


def _tone(freqs, dur=0.09, gap=0.03, vol=0.35) -> np.ndarray:
    parts = []
    for f in freqs:
        t = np.arange(int(SR * dur)) / SR
        env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)   # 10 ms ramps, no clicks
        parts.append(vol * env * np.sin(2 * np.pi * f * t))
        parts.append(np.zeros(int(SR * gap)))
    return np.concatenate(parts).astype(np.float32)


CUES = {
    "listening": _tone([660, 880]),            # rising: I am listening
    "capturing": _tone([1320], dur=0.05),      # shutter tick
    "thinking": _tone([440], dur=0.06),        # looped while a slow answer runs
    "done": _tone([880, 660]),                 # falling: finished
    "safety_paused": _tone([300, 300, 300], dur=0.12, gap=0.06, vol=0.5),
    "alert": _tone([1000, 1400], dur=0.07, gap=0.01, vol=0.6),
}


def samples(name: str) -> np.ndarray:
    return CUES[name]


def play(name: str, block: bool = False) -> None:
    data = CUES[name]
    from .demo_view import tap_audio
    tap_audio(data, SR)

    def _run():
        try:
            if not shutil.which("paplay"):
                return
            subprocess.run(["paplay", "--raw", f"--rate={SR}", "--channels=1",
                            "--format=float32le"], input=data.tobytes(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except Exception as e:  # noqa: BLE001
            print(f"[earcon] {name} failed: {e}")

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True, name=f"earcon-{name}").start()

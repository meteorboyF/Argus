"""Voice-activity endpointing (M6): stop recording when the user stops.

Replaces the fixed 5 s recording. Silero VAD (the ONNX model bundled with
faster-whisper, CPU) scores each 32 ms window; a small state machine with
hysteresis decides when speech starts and when the utterance is over:

  waiting  -> speaking   probability >= start_p for min_speech_s
  speaking -> done       probability <  end_p for silence_s (0.7 s)
  any      -> done       max_s reached (15 s cap)
  waiting  -> timeout    no speech within start_timeout_s

`on_speech(bool)` lets the speech arbiter hold non-safety speech while the
user talks (and cut an answer on barge-in).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

WINDOW = 512          # samples at 16 kHz = 32 ms (Silero's native window)
SR = 16000


@dataclass
class EndpointConfig:
    start_p: float = 0.5
    end_p: float = 0.35
    min_speech_s: float = 0.15
    silence_s: float = 0.7
    max_s: float = 15.0
    start_timeout_s: float = 5.0


class Endpointer:
    def __init__(self, prob_fn: Callable[[np.ndarray], float],
                 cfg: EndpointConfig = EndpointConfig(),
                 on_speech: Callable[[bool], None] | None = None):
        self.prob_fn, self.cfg = prob_fn, cfg
        self.on_speech = on_speech or (lambda s: None)
        self.state = "waiting"
        self.t = 0.0
        self._voiced = 0.0
        self._quiet = 0.0
        self.speech_start: float | None = None
        self._buf = np.zeros(0, np.float32)

    def feed(self, audio: np.ndarray) -> str:
        """Feed any number of samples; returns the state after them."""
        self._buf = np.concatenate([self._buf, audio.astype(np.float32)])
        dt = WINDOW / SR
        while len(self._buf) >= WINDOW and self.state in ("waiting", "speaking"):
            win, self._buf = self._buf[:WINDOW], self._buf[WINDOW:]
            p = float(self.prob_fn(win))
            self.t += dt
            if self.state == "waiting":
                self._voiced = self._voiced + dt if p >= self.cfg.start_p else 0.0
                if self._voiced >= self.cfg.min_speech_s:
                    self.state = "speaking"
                    self.speech_start = self.t - self._voiced
                    self.on_speech(True)
                elif self.t >= self.cfg.start_timeout_s:
                    self.state = "timeout"
            else:
                self._quiet = self._quiet + dt if p < self.cfg.end_p else 0.0
                if self._quiet >= self.cfg.silence_s or self.t >= self.cfg.max_s:
                    self.state = "done"
                    self.on_speech(False)
        return self.state


_MODEL = None


def silero_prob_fn() -> Callable[[np.ndarray], float]:
    """Speech probability for one 512-sample 16 kHz window."""
    global _MODEL
    if _MODEL is None:
        from faster_whisper.vad import get_vad_model
        _MODEL = get_vad_model()
    return lambda win: float(np.asarray(_MODEL(win.astype(np.float32))).reshape(-1)[-1])


def record_utterance(device=None, cfg: EndpointConfig = EndpointConfig(),
                     on_speech: Callable[[bool], None] | None = None,
                     pre_roll_s: float = 0.3) -> tuple[np.ndarray, str]:
    """Blocking: record from the mic until the endpointer finishes.
    Returns (float32 16 kHz audio trimmed to the utterance, final state)."""
    import sounddevice as sd
    from .speech import _resolve_device
    ep = Endpointer(silero_prob_fn(), cfg, on_speech)
    chunks: list[np.ndarray] = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=WINDOW,
                        device=_resolve_device(device, "input")) as stream:
        while ep.state in ("waiting", "speaking"):
            data, _ = stream.read(WINDOW)
            chunks.append(data[:, 0].copy())
            ep.feed(data[:, 0])
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    if ep.speech_start is not None:
        start = max(0, int((ep.speech_start - pre_roll_s) * SR))
        audio = audio[start:]
    return audio, ep.state

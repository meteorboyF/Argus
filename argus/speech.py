"""Speech I/O — all on CPU.

- WakeWord:   openWakeWord, listens continuously for the hotword.
- Transcriber: faster-whisper (tiny, INT8), transcribes a short recording.
- Speaker:    Piper TTS, synthesizes and plays a spoken reply.

Audio capture/playback uses sounddevice. Devices are selectable via
speech.input_device / speech.output_device in argus.yaml (index or name
substring — list with `python3 -m sounddevice`). On a headless bring-up without
audio hardware, construct Speaker with enabled=False to no-op safely.

openWakeWord usage note: the model keeps INTERNAL streaming state and must be
fed consecutive, non-overlapping 80 ms blocks (1280 samples at 16 kHz). Feeding
it a re-sent rolling window corrupts the stream and kills detection accuracy —
pass each mic block exactly once.
"""
from __future__ import annotations

import queue
import threading
import time
from enum import IntEnum

import numpy as np

from .config import SpeechConfig


def _sd():
    """Lazy sounddevice import — keeps `argus selftest` and headless runs from
    needing PortAudio until audio is actually used."""
    import sounddevice as sd
    return sd


def _resolve_device(selector, kind: str):
    """Turn an index / name-substring selector into a sounddevice device id.
    Returns None (system default) if the selector is None or nothing matches."""
    if selector is None:
        return None
    if isinstance(selector, int):
        return selector
    if isinstance(selector, str) and selector.strip().lstrip("-").isdigit():
        return int(selector)
    import sounddevice as sd
    want = str(selector).lower()
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, dev in enumerate(sd.query_devices()):
        if want in dev["name"].lower() and dev[key] > 0:
            return i
    print(f"[speech] no {kind} device matching {selector!r}; using system default")
    return None


class WakeWord:
    def __init__(self, cfg: SpeechConfig):
        self.cfg = cfg
        from openwakeword.model import Model
        try:
            import openwakeword
            openwakeword.utils.download_models()
        except Exception:  # noqa: BLE001 — already downloaded / offline
            pass
        # If wake_model is a path to a custom model, load just that; otherwise
        # load the pretrained pack and match the requested name against scores.
        if cfg.wake_model.endswith((".onnx", ".tflite")):
            self.model = Model(wakeword_models=[cfg.wake_model])
        else:
            self.model = Model()
        self.target = cfg.wake_model.rsplit("/", 1)[-1].split(".")[0].lower()

    def detected(self, audio_int16: np.ndarray) -> bool:
        """Feed ONE new mic block (consecutive audio, not a rolling window)."""
        scores = self.model.predict(audio_int16)
        # Score keys are versioned (e.g. "hey_jarvis_v0.1"); match by prefix so a
        # configured name like "hey_jarvis" works. Never fall back to "any model
        # fired" — with the full pretrained pack loaded that means constant
        # false wakes from unrelated hotwords.
        for key, score in scores.items():
            if key.lower().startswith(self.target) and score >= self.cfg.wake_threshold:
                return True
        return False

    def reset(self):
        """Clear streaming state (call after handling a query)."""
        try:
            self.model.reset()
        except Exception:  # noqa: BLE001 — older openwakeword lacks reset()
            pass


class Transcriber:
    def __init__(self, cfg: SpeechConfig):
        self.cfg = cfg
        from faster_whisper import WhisperModel
        self.model = WhisperModel(cfg.whisper_model, device="cpu", compute_type=cfg.whisper_compute)

    def transcribe(self, audio_float32: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio_float32, language="en")
        return " ".join(s.text for s in segments).strip()


class Priority(IntEnum):
    """Speech urgency. Higher wins the speaker.

    NORMAL — agent answers, acknowledgements. Never interrupts anything.
    WARN   — advisory hazard. Jumps ahead of queued NORMAL speech but lets an
             in-flight utterance finish.
    DANGER — urgent hazard. Preempts whatever is playing, immediately.
    """
    NORMAL = 0
    WARN = 1
    DANGER = 2


# A safety phrase that has waited longer than this is describing a world that
# has moved on. Speaking it late is worse than not speaking it: the user gets a
# warning about a hazard they have already walked past.
SAFETY_MAX_AGE_S = 1.5


class Speaker:
    """Non-blocking, priority-aware TTS.

    `speak()` returns immediately — it must, because the fast safety loop calls
    it. Synthesis and playback happen on a dedicated thread, so the loop keeps
    capturing frames and evaluating hazards while ARGUS is talking.

    Priority ordering is the point of this class. A DANGER warning preempts an
    in-flight agent answer (`sd.stop()`) and discards queued NORMAL speech; a
    hazard warning must never wait behind a sentence about where the user's keys
    are. Stale safety phrases are dropped rather than spoken late.
    """

    def __init__(self, cfg: SpeechConfig, enabled: bool = True):
        self.cfg = cfg
        self.enabled = enabled
        self._voice = None
        if enabled:
            try:
                from piper import PiperVoice
                self._voice = PiperVoice.load(cfg.piper_voice)
            except Exception as e:  # noqa: BLE001
                print(f"[speech] Piper load failed: {e}")
                self.enabled = False

        self._cv = threading.Condition()
        self._queue: list[tuple[Priority, float, int, str]] = []
        self._seq = 0
        self._playing: Priority | None = None
        self._stopping = False
        self._idle = threading.Event()
        self._idle.set()
        self._thread = threading.Thread(target=self._run, name="argus-speaker", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- public
    def speak(self, text: str, priority: Priority = Priority.NORMAL):
        """Queue `text`. Never blocks. DANGER preempts whatever is playing."""
        if not text:
            return
        if not self.enabled or self._voice is None:
            print(f"[ARGUS says] {text}")
            return
        with self._cv:
            if priority >= Priority.DANGER:
                # Queued chit-chat is worthless next to an imminent collision.
                self._queue = [it for it in self._queue if it[0] >= Priority.DANGER]
                if self._playing is not None and self._playing < Priority.DANGER:
                    self._abort_playback()   # aborts the sd.wait() in _play
            self._seq += 1
            # Sort key: highest priority first, then FIFO within a priority.
            self._queue.append((priority, time.monotonic(), self._seq, text))
            self._queue.sort(key=lambda it: (-it[0], it[2]))
            self._idle.clear()
            self._cv.notify()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block until the queue drains. For one-shot `argus query` and tests."""
        return self._idle.wait(timeout)

    def stop(self):
        """Stop the playback thread, abandoning anything still queued."""
        with self._cv:
            self._stopping = True
            self._queue.clear()
            self._cv.notify_all()
        self._abort_playback()
        self._thread.join(timeout=2.0)

    def _abort_playback(self):
        """Best-effort cut-off of in-flight audio.

        Deliberately swallows everything: this runs on the fast safety loop's
        thread during DANGER preemption, and an audio-backend hiccup must not
        take the safety loop down with it.
        """
        try:
            _sd().stop()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- internal
    def _run(self):
        while True:
            with self._cv:
                while not self._queue and not self._stopping:
                    self._idle.set()
                    self._cv.wait(0.2)
                if self._stopping:
                    self._idle.set()
                    return
                priority, queued_at, _, text = self._queue.pop(0)
                self._playing = priority
            try:
                age = time.monotonic() - queued_at
                if priority >= Priority.WARN and age > SAFETY_MAX_AGE_S:
                    print(f"[speech] dropped stale {priority.name} ({age:.1f}s): {text}")
                else:
                    self._play(text)
            except Exception as e:  # noqa: BLE001 — never kill the speaker thread
                print(f"[speech] playback failed: {e}")
            finally:
                with self._cv:
                    self._playing = None
                    if not self._queue:
                        self._idle.set()

    def _play(self, text: str):
        # Current piper-tts API: synthesize() returns an iterable of AudioChunk
        # (one or more per sentence), each carrying its own float32 audio array
        # and sample rate — it no longer writes into a wave.Wave_write object.
        chunks = list(self._voice.synthesize(text))
        if not chunks:
            return
        sr = chunks[0].sample_rate
        data = (chunks[0].audio_float_array if len(chunks) == 1
                else np.concatenate([c.audio_float_array for c in chunks]))
        sd = _sd()
        sd.play(data, sr, device=_resolve_device(self.cfg.output_device, "output"))
        sd.wait()   # returns early if another thread calls sd.stop() (preemption)


def record(seconds: float, sample_rate: int, device=None) -> np.ndarray:
    """Blocking mic capture -> float32 mono in [-1, 1]."""
    import sounddevice as sd
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1,
                   dtype="float32", device=_resolve_device(device, "input"))
    sd.wait()
    return audio.flatten()


def mic_stream(sample_rate: int, block_ms: int = 80, device=None):
    """Generator yielding consecutive int16 audio blocks for wake-word listening.
    80 ms at 16 kHz = 1280 samples = exactly one openWakeWord frame."""
    import sounddevice as sd
    q: queue.Queue = queue.Queue()
    block = int(sample_rate * block_ms / 1000)

    def cb(indata, frames, time_info, status):  # noqa: ARG001
        q.put((indata[:, 0] * 32767).astype(np.int16).copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                        blocksize=block, device=_resolve_device(device, "input"),
                        callback=cb):
        while True:
            yield q.get()

"""Speech I/O — all on CPU.

- WakeWord:   openWakeWord, listens continuously for the hotword.
- Transcriber: faster-whisper (tiny, INT8), transcribes a short recording.
- Speaker:    Piper TTS, synthesizes and plays a spoken reply.

Audio capture/playback uses sounddevice. On a headless bring-up without audio
hardware, construct with enabled=False to no-op safely.
"""
from __future__ import annotations

import queue
import wave

import numpy as np

from .config import SpeechConfig


class WakeWord:
    def __init__(self, cfg: SpeechConfig):
        self.cfg = cfg
        from openwakeword.model import Model
        try:
            import openwakeword
            openwakeword.utils.download_models()
        except Exception:  # noqa: BLE001
            pass
        # If wake_model is a known name use wordlist; if it's a path, load it.
        if cfg.wake_model.endswith(".onnx"):
            self.model = Model(wakeword_models=[cfg.wake_model])
        else:
            self.model = Model()
        self.target = cfg.wake_model

    def detected(self, audio_int16: np.ndarray) -> bool:
        scores = self.model.predict(audio_int16)
        # Trigger if any model (or the named one) crosses threshold.
        if self.target in scores:
            return scores[self.target] >= self.cfg.wake_threshold
        return any(v >= self.cfg.wake_threshold for v in scores.values())


class Transcriber:
    def __init__(self, cfg: SpeechConfig):
        self.cfg = cfg
        from faster_whisper import WhisperModel
        self.model = WhisperModel(cfg.whisper_model, device="cpu", compute_type=cfg.whisper_compute)

    def transcribe(self, audio_float32: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio_float32, language="en")
        return " ".join(s.text for s in segments).strip()


class Speaker:
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

    def speak(self, text: str):
        if not self.enabled or self._voice is None or not text:
            print(f"[ARGUS says] {text}")
            return
        import io
        import sounddevice as sd
        import soundfile as sf
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize(text, wf)
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        sd.play(data, sr)
        sd.wait()


def record(seconds: float, sample_rate: int) -> np.ndarray:
    """Blocking mic capture -> float32 mono in [-1, 1]."""
    import sounddevice as sd
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def mic_stream(sample_rate: int, block_ms: int = 80):
    """Generator yielding int16 audio blocks for continuous wake-word listening."""
    import sounddevice as sd
    q: queue.Queue = queue.Queue()
    block = int(sample_rate * block_ms / 1000)

    def cb(indata, frames, time_info, status):  # noqa: ARG001
        q.put((indata[:, 0] * 32767).astype(np.int16).copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                        blocksize=block, callback=cb):
        while True:
            yield q.get()

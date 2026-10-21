"""Voice-activity endpointing: state machine plus a real Silero check on
speech synthesised by Piper (skipped when the voice model is absent)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from argus.vad import SR, WINDOW, EndpointConfig, Endpointer


def _probs(seq):
    it = iter(seq)
    return lambda win: next(it)


def _feed(ep, n_windows):
    return ep.feed(np.zeros(n_windows * WINDOW, np.float32))


def test_speech_then_silence_ends_after_0_7s():
    events = []
    # 0.32 s silence, 1 s speech, then silence.
    probs = [0.1] * 10 + [0.9] * 31 + [0.05] * 40
    ep = Endpointer(_probs(probs), EndpointConfig(), events.append)
    state = _feed(ep, len(probs))
    assert state == "done" and events == [True, False]
    # done ~0.7 s after the last voiced window
    assert abs(ep.t - (41 * WINDOW / SR + 0.7)) < 0.05


def test_short_pause_does_not_end_the_utterance():
    probs = [0.9] * 20 + [0.1] * 10 + [0.9] * 20 + [0.1] * 25
    ep = Endpointer(_probs(probs))
    _feed(ep, 30)
    assert ep.state == "speaking"            # 0.32 s pause < 0.7 s
    _feed(ep, 45)
    assert ep.state == "done"


def test_no_speech_times_out():
    ep = Endpointer(_probs([0.05] * 400), EndpointConfig(start_timeout_s=2.0))
    assert _feed(ep, 400) == "timeout"


def test_cap_at_max_seconds():
    ep = Endpointer(_probs([0.95] * 1000), EndpointConfig(max_s=3.0))
    assert _feed(ep, 1000) == "done" and ep.t <= 3.05


def test_single_click_is_not_speech():
    ep = Endpointer(_probs([0.1] * 5 + [0.95] + [0.1] * 200),
                    EndpointConfig(start_timeout_s=3.0))
    assert _feed(ep, 206) == "timeout"


VOICE = Path("/opt/argus/models/piper/en_US-lessac-medium.onnx")


@pytest.mark.skipif(not VOICE.exists(), reason="Piper voice not installed")
def test_silero_detects_piper_speech_and_endpoints():
    from piper import PiperVoice
    from argus.vad import silero_prob_fn
    v = PiperVoice.load(str(VOICE))
    chunks = list(v.synthesize("Hey Argus, what is in front of me?"))
    sr0 = chunks[0].sample_rate
    speech = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    n = int(len(speech) * SR / sr0)
    speech16 = np.interp(np.linspace(0, len(speech) - 1, n), np.arange(len(speech)), speech)
    audio = np.concatenate([np.zeros(SR // 2), speech16, np.zeros(SR * 2)]).astype(np.float32)
    events = []
    ep = Endpointer(silero_prob_fn(), EndpointConfig(), events.append)
    ep.feed(audio)
    assert events == [True, False] and ep.state == "done"
    speech_end = 0.5 + n / SR
    # Starts near the speech onset; ends ~0.7 s after the voice stops. The
    # Piper clip carries ~0.2 s of its own trailing silence, hence +0.3.
    assert 0.3 < ep.speech_start < 0.9
    assert speech_end + 0.3 < ep.t < speech_end + 1.2

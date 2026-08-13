"""Speaker priority + preemption tests.

These run anywhere — no Piper, no PortAudio, no Jetson. The real `_play` is
replaced with a recorder so the queueing and preemption logic can be tested on
its own, which is the part that matters for safety.

What is being protected here: a DANGER warning must never wait behind an agent
answer, and `speak()` must never block its caller (the fast safety loop calls
it, and a blocked fast loop is a blind fast loop).
"""
from __future__ import annotations

import threading
import time

import pytest

from argus.config import SpeechConfig
from argus.speech import Priority, Speaker


class RecordingSpeaker(Speaker):
    """Speaker whose playback is a recorded sleep instead of real audio."""

    def __init__(self, play_seconds: float = 0.05):
        self._play_seconds = play_seconds
        self.played: list[str] = []
        self.interrupted: list[str] = []
        self._interrupt = threading.Event()
        # enabled=False so __init__ skips the Piper load, then re-arm the fields
        # the playback thread needs. The thread is already running and blocked
        # on an empty queue, so this is safe.
        super().__init__(SpeechConfig(), enabled=False)
        self.enabled = True
        self._voice = object()

    def _play(self, text: str):
        self.played.append(text)
        # Simulate playback that can be cut short by a preempting DANGER.
        if self._interrupt.wait(self._play_seconds):
            self._interrupt.clear()
            self.interrupted.append(text)

    def speak(self, text: str, priority: Priority = Priority.NORMAL):
        # Mirror the real preemption side effect: sd.stop() aborts sd.wait().
        if priority >= Priority.DANGER:
            self._interrupt.set()
        super().speak(text, priority)


@pytest.fixture
def spk():
    s = RecordingSpeaker()
    yield s
    s.stop()


def test_speak_does_not_block_the_caller(spk):
    """The fast loop calls speak(); it must return far faster than playback."""
    spk._play_seconds = 1.0
    t0 = time.perf_counter()
    spk.speak("a long spoken sentence", Priority.NORMAL)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, f"speak() blocked for {elapsed:.3f}s"


def test_danger_preempts_an_in_flight_normal_utterance(spk):
    spk._play_seconds = 2.0
    spk.speak("your keys are on the table to your right", Priority.NORMAL)
    time.sleep(0.1)                      # let playback actually start
    spk.speak("Stop. Obstacle very close.", Priority.DANGER)
    assert spk.wait_until_idle(timeout=5.0)
    assert "your keys are on the table to your right" in spk.interrupted
    assert "Stop. Obstacle very close." in spk.played


def test_danger_discards_queued_normal_speech(spk):
    spk._play_seconds = 0.4
    spk.speak("first answer", Priority.NORMAL)
    time.sleep(0.05)
    spk.speak("second answer", Priority.NORMAL)   # still queued, not playing
    spk.speak("Stop. Step down ahead.", Priority.DANGER)
    assert spk.wait_until_idle(timeout=5.0)
    assert "Stop. Step down ahead." in spk.played
    assert "second answer" not in spk.played, "stale chit-chat survived a DANGER"


def test_warn_jumps_ahead_of_queued_normal_speech(spk):
    spk._play_seconds = 0.3
    spk.speak("playing now", Priority.NORMAL)
    time.sleep(0.05)
    spk.speak("queued answer", Priority.NORMAL)
    spk.speak("Careful, obstacle ahead.", Priority.WARN)
    assert spk.wait_until_idle(timeout=5.0)
    # WARN does not interrupt, but it must be spoken before the queued NORMAL.
    assert spk.played.index("Careful, obstacle ahead.") < spk.played.index("queued answer")


def test_stale_safety_phrases_are_dropped_not_spoken_late(spk, monkeypatch):
    """A warning about a hazard the user already walked past is worse than none.

    The WARN has to sit in the queue to go stale, so a long NORMAL is put in
    front of it (WARN outranks queued NORMAL but does not interrupt playback).
    """
    monkeypatch.setattr("argus.speech.SAFETY_MAX_AGE_S", 0.1)
    spk._play_seconds = 0.5
    spk.speak("a long agent answer", Priority.NORMAL)
    time.sleep(0.05)                     # ensure the NORMAL is playing
    spk.speak("Careful, obstacle ahead.", Priority.WARN)
    assert spk.wait_until_idle(timeout=5.0)
    assert "a long agent answer" in spk.played
    assert "Careful, obstacle ahead." not in spk.played


def test_normal_speech_is_fifo(spk):
    spk._play_seconds = 0.02
    for i in range(5):
        spk.speak(f"line {i}", Priority.NORMAL)
    assert spk.wait_until_idle(timeout=5.0)
    assert spk.played == [f"line {i}" for i in range(5)]

"""Speech arbiter policy (M6): P0-P4 preemption, resume, barge-in, watchdog."""
from __future__ import annotations

import threading
import time

import pytest

from argus.arbiter import REPEAT_OFFER, Arbiter
from argus.speech import Priority


class FakePlayer:
    def __init__(self, seconds=0.3):
        self.seconds = seconds
        self.played: list[str] = []
        self.aborted: list[str] = []
        self.abort_latency: list[float] = []
        self._abort = threading.Event()
        self._abort_called_at = None
        self.current = None

    def play(self, text):
        self.current = text
        self.played.append(text)
        self._abort.clear()
        if self._abort.wait(self.seconds):
            self.aborted.append(text)
            self.abort_latency.append(time.perf_counter() - self._abort_called_at)
        self.current = None

    def abort(self):
        self._abort_called_at = time.perf_counter()
        self._abort.set()


class Clock:
    def __init__(self):
        self.offset = 0.0

    def __call__(self):
        return time.monotonic() + self.offset


@pytest.fixture
def rig():
    p, c = FakePlayer(), Clock()
    earcons = []
    a = Arbiter(p.play, p.abort, earcon=earcons.append, clock=c)
    yield a, p, c, earcons
    a.stop()


def _wait_playing(p, text, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end and p.current != text:
        time.sleep(0.005)
    assert p.current == text


def test_p0_cuts_an_answer_within_50ms(rig):
    a, p, _, _ = rig
    p.seconds = 2.0
    a.say("your keys are on the table", Priority.NORMAL)
    _wait_playing(p, "your keys are on the table")
    a.say("Stop. Left.", Priority.DANGER)
    assert a.wait_until_idle(5)
    assert "your keys are on the table" in p.aborted
    assert p.abort_latency[0] < 0.05
    assert p.played.index("Stop. Left.") > 0


def test_p1_cuts_guidance_but_not_danger(rig):
    a, p, _, _ = rig
    p.seconds = 1.0
    a.say("Seat, slightly right, four steps.", Priority.GUIDANCE)
    _wait_playing(p, "Seat, slightly right, four steps.")
    a.say("Careful. Right.", Priority.WARN)
    _wait_playing(p, "Careful. Right.")
    assert "Seat, slightly right, four steps." in p.aborted
    # A WARN must not cut a DANGER.
    a.say("Stop. Drop ahead.", Priority.DANGER)
    _wait_playing(p, "Stop. Drop ahead.")
    a.say("Careful. Left.", Priority.WARN)
    time.sleep(0.1)
    assert "Stop. Drop ahead." not in p.aborted


def test_interrupted_answer_resumes_when_fresh(rig):
    a, p, _, _ = rig
    p.seconds = 0.4
    a.say("the label says take one tablet daily", Priority.NORMAL)
    _wait_playing(p, "the label says take one tablet daily")
    a.say("Stop.", Priority.DANGER)
    assert a.wait_until_idle(5)
    assert p.played == ["the label says take one tablet daily", "Stop.",
                        "the label says take one tablet daily"]


def test_interrupted_answer_older_than_5s_offers_repeat(rig):
    a, p, c, _ = rig
    p.seconds = 0.4
    a.say("a long description of the room", Priority.NORMAL)
    _wait_playing(p, "a long description of the room")
    a.say("Stop.", Priority.DANGER)
    _wait_playing(p, "Stop.")
    c.offset += 6.0            # the hazard took long: the answer is now stale
    assert a.wait_until_idle(5)
    assert p.played[-1] == REPEAT_OFFER
    a.repeat()
    assert a.wait_until_idle(5)
    assert p.played[-1] == "a long description of the room"


def test_user_speaking_holds_p2_to_p4_but_not_hazards(rig):
    a, p, _, _ = rig
    p.seconds = 0.05
    a.set_user_speaking(True)
    a.say("an answer", Priority.NORMAL)
    a.say("Careful. Ahead.", Priority.WARN)
    time.sleep(0.3)
    assert p.played == ["Careful. Ahead."]
    a.set_user_speaking(False)
    assert a.wait_until_idle(5)
    assert p.played == ["Careful. Ahead.", "an answer"]


def test_barge_in_stops_an_answer(rig):
    a, p, _, _ = rig
    p.seconds = 2.0
    a.say("a very long answer", Priority.NORMAL)
    _wait_playing(p, "a very long answer")
    a.set_user_speaking(True)
    time.sleep(0.1)
    assert "a very long answer" in p.aborted


def test_stale_safety_is_dropped(rig):
    a, p, c, _ = rig
    p.seconds = 0.05
    a.set_user_speaking(True)            # nothing else will play meanwhile...
    a.say("old hazard", Priority.WARN)   # ...but WARN is never held: it plays now
    assert a.wait_until_idle(2)
    assert "old hazard" in p.played
    # A WARN that sits behind a DANGER long enough goes stale.
    p.seconds = 0.3
    a.set_user_speaking(False)
    a.say("Stop.", Priority.DANGER)
    a.say("Careful.", Priority.WARN)
    _wait_playing(p, "Stop.")
    c.offset += 2.0
    assert a.wait_until_idle(5)
    assert "Careful." not in p.played
    assert any(e[0] == "stale_drop" and e[1] == "Careful." for e in a.log)


def test_watchdog_earcon_when_depth_goes_stale(rig):
    a, _, _, earcons = rig
    a.depth_heartbeat()
    time.sleep(0.1)
    assert earcons == []
    time.sleep(0.4)                      # no heartbeat for > 250 ms
    assert earcons == ["safety_paused"]
    a.depth_heartbeat()
    time.sleep(0.1)
    assert any(e[0] == "safety_resumed" for e in a.log)


def test_priority_order():
    assert (Priority.SYSTEM < Priority.NORMAL < Priority.GUIDANCE
            < Priority.WARN < Priority.DANGER)

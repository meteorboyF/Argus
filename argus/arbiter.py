"""Speech arbiter (M6): who may speak, when, and what happens to the loser.

Policy ("the fast loop inhibits slow-loop audio"):

  P0 DANGER   cuts whatever is playing at once (target < 50 ms) and drops
              queued non-safety speech.
  P1 WARN     cuts GUIDANCE / NORMAL / SYSTEM speech (target < 300 ms).
  P2 GUIDANCE, P3 NORMAL (answers), P4 SYSTEM queue behind safety.

- A safety phrase older than SAFETY_MAX_AGE_S is dropped, never spoken late.
- An answer (P3) cut off by a hazard resumes if it is under RESUME_MAX_AGE_S
  old; otherwise ARGUS offers "Want me to repeat that?" and `repeat()` says it.
- While the user is speaking (voice activity), P2-P4 wait; P0/P1 never wait.
- Watchdog: if the depth loop stops publishing for DEPTH_STALE_S, the
  "safety paused" earcon plays once and a SYSTEM line is logged; it re-arms
  when depth returns.

The arbiter is speaker-agnostic: it drives a blocking `play(text)` and an
`abort()` that makes `play` return early. `ArbitratedSpeaker` adapts the
existing Piper `Speaker` and keeps its public API (speak / busy /
wait_until_idle / cancel_normal / stop).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .speech import Priority, SAFETY_MAX_AGE_S
from .telemetry import bus

RESUME_MAX_AGE_S = 5.0
DEPTH_STALE_S = 0.25
REPEAT_OFFER = "Want me to repeat that?"


@dataclass(order=True)
class _Item:
    sort_key: tuple
    text: str = field(compare=False)
    priority: Priority = field(compare=False)
    queued_at: float = field(compare=False)


class Arbiter:
    def __init__(self, play: Callable[[str], None], abort: Callable[[], None],
                 earcon: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._play_fn, self._abort_fn = play, abort
        self._earcon = earcon or (lambda name: None)
        self._clock = clock
        self._cv = threading.Condition()
        self._queue: list[_Item] = []
        self._seq = 0
        self._current: _Item | None = None
        self._current_started = 0.0
        self._aborted_current = False
        self._interrupted: tuple[str, float] | None = None   # (answer text, when cut)
        self.last_answer = ""
        self._user_speaking = False
        self._stopping = False
        self._idle = threading.Event()
        self._idle.set()
        self.log: list[tuple[str, str, float]] = []           # (event, text, t)
        self._last_depth = None
        self._paused_announced = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="arbiter")
        self._thread.start()
        self._wd = threading.Thread(target=self._watchdog, daemon=True, name="arbiter-wd")
        self._wd.start()

    # ------------------------------------------------------------------ API
    def say(self, text: str, priority: Priority = Priority.NORMAL) -> None:
        if not text:
            return
        now = self._clock()
        with self._cv:
            if priority >= Priority.DANGER:
                self._queue = [i for i in self._queue if i.priority >= Priority.WARN]
            self._seq += 1
            self._queue.append(_Item((-int(priority), self._seq), text, priority, now))
            self._queue.sort()
            self._idle.clear()
            cur = self._current
            if cur is not None and self._preempts(priority, cur.priority):
                self._aborted_current = True
                self._abort_fn()
                self.log.append(("preempt", cur.text, now))
            self._cv.notify_all()

    @staticmethod
    def _preempts(new: Priority, playing: Priority) -> bool:
        if new >= Priority.DANGER:
            return playing < Priority.DANGER
        if new >= Priority.WARN:
            return playing < Priority.WARN
        return False

    def set_user_speaking(self, speaking: bool) -> None:
        with self._cv:
            self._user_speaking = speaking
            if speaking and self._current is not None and self._current.priority < Priority.WARN:
                # Barge-in: the user talking over an answer stops the answer.
                self._aborted_current = True
                self._abort_fn()
            self._cv.notify_all()

    def repeat(self) -> None:
        if self.last_answer:
            self.say(self.last_answer, Priority.NORMAL)

    def cancel_normal(self) -> None:
        with self._cv:
            self._queue = [i for i in self._queue if i.priority >= Priority.WARN]
            if self._current is not None and self._current.priority < Priority.WARN:
                self._aborted_current = True
                self._abort_fn()

    def depth_heartbeat(self) -> None:
        self._last_depth = self._clock()

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def stop(self) -> None:
        with self._cv:
            self._stopping = True
            self._queue.clear()
            self._cv.notify_all()
        self._abort_fn()
        self._thread.join(timeout=2.0)
        self._wd.join(timeout=1.0)

    # ------------------------------------------------------------------ worker
    def _next(self) -> _Item | None:
        """Highest-priority eligible item; drops stale safety phrases."""
        now = self._clock()
        keep = []
        for it in self._queue:
            if it.priority >= Priority.WARN and now - it.queued_at > SAFETY_MAX_AGE_S:
                self.log.append(("stale_drop", it.text, now))
                continue
            keep.append(it)
        self._queue = keep
        for i, it in enumerate(self._queue):
            if self._user_speaking and it.priority < Priority.WARN:
                continue
            return self._queue.pop(i)
        return None

    def _run(self) -> None:
        while True:
            with self._cv:
                item = None
                while not self._stopping:
                    item = self._next()
                    if item is not None:
                        break
                    if not self._queue:
                        self._idle.set()
                    self._cv.wait(0.05)
                if self._stopping:
                    self._idle.set()
                    return
                self._current, self._aborted_current = item, False
                self._current_started = self._clock()
            self.log.append(("start", item.text, self._clock()))
            try:
                self._play_fn(item.text)
            except Exception as e:  # noqa: BLE001 — never kill the arbiter thread
                print(f"[arbiter] playback failed: {e}")
            with self._cv:
                aborted = self._aborted_current
                self._current = None
                now = self._clock()
                if item.priority == Priority.NORMAL:
                    self.last_answer = item.text
                    if aborted:
                        self._interrupted = (item.text, now)
                        self.log.append(("interrupted", item.text, now))
                    elif self._interrupted and self._interrupted[0] == item.text:
                        self._interrupted = None
                self.log.append(("aborted" if aborted else "done", item.text, now))
                # After a safety phrase finishes, decide what happens to a cut answer.
                if (item.priority >= Priority.WARN and self._interrupted is not None
                        and not any(i.priority >= Priority.WARN for i in self._queue)):
                    text, cut_at = self._interrupted
                    self._interrupted = None
                    self._seq += 1
                    if now - cut_at <= RESUME_MAX_AGE_S:
                        self._queue.insert(0, _Item((-int(Priority.NORMAL), -self._seq), text,
                                                    Priority.NORMAL, now))
                    else:
                        self._queue.append(_Item((-int(Priority.SYSTEM), self._seq),
                                                 REPEAT_OFFER, Priority.SYSTEM, now))
                    self._queue.sort()

    def _watchdog(self) -> None:
        while not self._stopping:
            time.sleep(0.05)
            last = self._last_depth
            if last is None:
                continue
            stale = self._clock() - last > DEPTH_STALE_S
            if stale and not self._paused_announced:
                self._paused_announced = True
                self._earcon("safety_paused")
                bus.log("system", "safety paused: no fresh depth")
                self.log.append(("safety_paused", "", self._clock()))
            elif not stale and self._paused_announced:
                self._paused_announced = False
                self.log.append(("safety_resumed", "", self._clock()))


class ArbitratedSpeaker:
    """Drop-in for `Speaker` with the arbiter's P0-P4 policy."""

    def __init__(self, speaker):
        from . import earcons
        self.speaker = speaker
        self.enabled = speaker.enabled
        self.arbiter = Arbiter(play=self._play, abort=speaker._abort_playback,
                               earcon=earcons.play)

    def _play(self, text: str) -> None:
        if self.speaker.enabled and self.speaker._voice is not None:
            self.speaker._play(text)

    def speak(self, text: str, priority: Priority = Priority.NORMAL) -> None:
        if text:
            bus.log("safety" if priority >= Priority.WARN else "argus", text)
        self.arbiter.say(text, priority)

    @property
    def busy(self) -> bool:
        return self.arbiter.busy

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        return self.arbiter.wait_until_idle(timeout)

    def cancel_normal(self) -> None:
        self.arbiter.cancel_normal()

    def stop(self) -> None:
        self.arbiter.stop()
        self.speaker.stop()

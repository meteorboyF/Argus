"""Bounded in-session timers/reminders. No persistence or external messages."""
from __future__ import annotations
import heapq
import re
import threading
import time
from .sensitive import mask_text

_NUMBERS = dict(zip(('one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','fifteen','twenty','thirty','forty','sixty'),(1,2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,60)))
_PATTERN = re.compile(r'^(?:set (?:a |an )?timer for|remind me in) (\d+|'+ '|'.join(_NUMBERS)+r') (seconds?|minutes?|hours?)(?: to (.+))?[.!?]*$',re.I)


def parse_reminder(text):
    match = _PATTERN.fullmatch(text.strip())
    if not match:
        return None
    amount = int(match[1]) if match[1].isdigit() else _NUMBERS[match[1].lower()]
    seconds = amount * (3600 if match[2].lower().startswith('hour') else 60 if match[2].lower().startswith('minute') else 1)
    if not 1 <= seconds <= 86400:
        return None
    return seconds,mask_text((match[3] or 'Your timer is ready').strip())[:200]


class Reminders:
    def __init__(self, announce, clock=time.monotonic, start=True):
        self.announce, self.clock = announce,clock
        self._items, self._sequence = [],0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if start:
            self._thread = threading.Thread(target=self._run,name='argus-reminders',daemon=True)
            self._thread.start()

    def add(self, seconds, message):
        if not 1 <= seconds <= 86400:
            return False
        with self._lock:
            if len(self._items) >= 16 or self._stop.is_set():
                return False
            self._sequence += 1
            heapq.heappush(self._items,(self.clock()+seconds,self._sequence,mask_text(message)[:200]))
        return True

    def clear(self):
        with self._lock:
            self._items.clear()

    def due(self):
        items = []
        with self._lock:
            while self._items and self._items[0][0] <= self.clock():
                items.append(heapq.heappop(self._items)[2])
        return items

    def _run(self):
        while not self._stop.wait(.2):
            for message in self.due():
                self.announce('Reminder. '+message)

    def stop(self):
        self._stop.set()
        self.clear()
        if self._thread is not None:
            self._thread.join(timeout=1)

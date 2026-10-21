"""Bounded, session-only conversational context and deterministic assistance.

No disk, face identity, inferred location, or unobserved action is remembered.
"""
from __future__ import annotations

import collections
import re
import threading
import time
from dataclasses import dataclass
from .sensitive import mask_text


@dataclass(frozen=True)
class Observation:
    name: str
    found: bool
    direction: str
    at: float


class SessionMemory:
    def __init__(self, limit=32, current_s=8, remember_s=300, clock=time.monotonic):
        self.clock, self.current_s, self.remember_s = clock, current_s, remember_s
        self._items = collections.OrderedDict()
        self.limit = limit
        self.target = ''
        self.preferences = {}
        self._lock = threading.RLock()

    def observe(self, name, found, direction=''):
        name = mask_text(name.strip().lower())[:80]
        with self._lock:
            self.target = name
            self._items[name] = Observation(name, bool(found), direction if found else '', self.clock())
            self._items.move_to_end(name)
            while len(self._items) > self.limit:
                self._items.popitem(last=False)

    def last(self, name=None):
        with self._lock:
            item = self._items.get(name or self.target)
            if item is None or self.clock()-item.at > self.remember_s:
                return None
            return item

    def report(self, name=None):
        item = self.last(name)
        if item is None:
            return "I don't have a recent observation of that. Ask me to look again."
        age = self.clock()-item.at
        if not item.found:
            return f"I did not find the {item.name} in the last view. That doesn't mean it isn't nearby."
        if age <= self.current_s:
            return f"In the last view, the {item.name} was {item.direction}."
        return f"I last saw the {item.name} {item.direction}, {int(age)} seconds ago. Ask me to check again."

    def forget(self):
        with self._lock:
            self._items.clear()
            self.target = ''
            self.preferences.clear()

    def context(self):
        with self._lock:
            items = [i for i in self._items.values() if self.clock()-i.at <= self.remember_s]
        preference = self.preferences.get('detail','brief')
        name = self.preferences.get('name','not provided')
        return f'User prefers {preference} replies; preferred name: {name}.\n'+'\n'.join(self.report(i.name) for i in items[-4:])


def companion_route(text, memory):
    """Return (skill, name, spoken answer), or None for the normal router."""
    t = text.strip().lower().strip('?.!')
    if re.fullmatch(r'(what objects (can you (see|detect)|are (here|around|nearby))|list (the )?objects|scan (the )?objects)',t):
        return ('inventory','','')
    match = re.fullmatch(r'(?:my name is|call me) (.+)',t)
    if match:
        return ('preference_name',mask_text(match[1])[:40],'')
    if re.fullmatch(r"(what is|what's) my name",t):
        name = memory.preferences.get('name')
        return ('answer','',f'You asked me to call you {name}.' if name else 'What would you like me to call you?')
    if re.fullmatch(r'(what did we (talk about|discuss)|summari[sz]e our conversation|recap)',t):
        return ('recap','','')
    if re.match(r'^(call|phone|text|email|message|buy|order|book a)\b',t):
        return ('answer','','I cannot place calls, send messages or make purchases. I can help with what I can observe and the tools available on this rig.')
    from .reminders import parse_reminder
    reminder = parse_reminder(t)
    if reminder is not None:
        import json
        return ('reminder',json.dumps(reminder),'')
    if t.startswith(('set a timer','set timer','remind me')):
        return ('answer','','Use a whole number of seconds, minutes or hours, up to one day. Reminders last only for this session.')
    if re.fullmatch(r'(cancel|clear|stop) (all )?(timers|reminders)',t):
        return ('clear_reminders','','')
    if re.fullmatch(r'(be brief|shorter answers|keep it short)',t):
        return ('preference','brief','I will keep replies brief for this session.')
    if re.fullmatch(r'(more detail|longer answers|explain more)',t):
        return ('preference','detail','I will give a little more detail for this session.')
    if re.fullmatch(r'(forget|clear|erase) (everything|our conversation|my history|your memory)', t):
        return ('forget', '', '')
    if re.fullmatch(r'(what can you do|help|help me|what are your capabilities)', t):
        return ('answer', '', "I can talk with you, find or watch objects, read pages, and help you find a seat. With calibrated tracking, I can remember places and plan a local route; walking guidance still needs acceptance testing.")
    if re.fullmatch(r'(are you there|are you listening|hello|hi|hey)', t):
        return ('answer', '', "I'm here. What would you like help with?")
    if re.fullmatch(r'(what (are|were) you doing|what is the current task|status)', t):
        return ('status', '', '')
    if re.fullmatch(r'(is (it|that) still there|where is (it|that)( now)?|find (it|that)( again)?|look for (it|that)( again)?)', t):
        return ('find', memory.target, '') if memory.target else ('answer', '', 'Which object should I look for?')
    if re.fullmatch(r'(where did you (last )?see (it|that)|what did you find|what did you last see)', t):
        return ('answer', '', memory.report())
    match = re.fullmatch(r'(?:watch|keep watching|keep looking for|search for|look for) (?:the |my |a |an )?(.+)', t)
    if match:
        name = match[1].strip()
        if name in ('it', 'that'):
            name = memory.target
        return ('watch', name, '') if name else ('answer', '', 'Which object should I watch for?')
    match = re.fullmatch(r'(?:remember|mark|save) (?:this (?:place|spot|location)|here) as (.+)', t)
    if match:
        return ('mark_place', mask_text(match[1])[:60], '')
    if re.fullmatch(r'(what places (do you|have you) (remember|saved)|list (my )?places)', t):
        return ('places', '', '')
    match = re.fullmatch(r'(?:navigate|guide me|take me|go|return|go back) to (?:the |a |an )?(.+)', t)
    if match:
        name = mask_text(match[1])[:60]
        if name in ('chair','seat','sofa','bench'):
            return ('seat',name,'Let me look for a seat.')
        return ('navigate', name, '')
    if re.fullmatch(r'(where am i|how is (your )?tracking|tracking status|map status)', t):
        return ('location', '', '')
    if (re.search(r'\b(is (it|the path|the way) (safe|clear)|can i (cross|walk|go)|safe to (cross|walk))\b', t)
            or re.search(r'\b(safe|clear)\b',t) and re.search(r'\b(path|walk|cross|way|road|street)\b',t)):
        return ('safety_status', '', '')
    return None

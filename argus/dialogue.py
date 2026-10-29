"""Conversation layer: ARGUS as a spoken assistant that gets things done.

"Gemma talks; the skills do the work."

    utterance
      -> rule router (instant, deterministic): stop / pause / repeat / next
         page / read / seat / find / describe / page questions
      -> otherwise Gemma router (one text-only call): returns
         {"skill": ..., "name": ..., "say": ...}, validated against the skill
         list; for small talk "say" IS the answer
      -> skill: reading coach + OCR, seat guidance, grounding, privacy-gated
         scene description, text-only answers about the page, chat

Why the split: on this rig Gemma 4 E2B misread text in images and ignored
instructions to call tools (DECISION_LOG 2026-08-14, 2026-10-02). It is good
at understanding requests and phrasing replies, so it routes and talks;
deterministic code reads, measures and decides anything safety-relevant.
The fast safety loop is untouched: hazard speech still preempts everything.

Context: the last few exchanges and the active task ("reading page 3") go
into every prompt, so "next page", "read that again" and "what did it say
about the dose" work as follow-ups.
"""
from __future__ import annotations

import collections
import json
import re
import time
from dataclasses import dataclass, field

from .telemetry import bus

PERSONA = (
    "You are ARGUS, a calm, warm and capable assistant built into smart glasses "
    "worn by a blind person. Talk like a helpful, relaxed person, not a formal "
    "service desk. Answer the user's actual question directly. For advice, give "
    "a useful concrete suggestion first; ask a follow-up only if needed. "
    "Use the conversation to understand follow-ups; do not repeat greetings, "
    "capability pitches, or filler such as 'I am functioning well'. Speak naturally, "
    "in at most two short sentences. Use plain words, no lists, no markdown, no "
    "emoji. Acknowledge actions only when using a skill. Never pretend to see something you have "
    "not been shown; never guess private numbers. Never promise protection or "
    "safe walking. Missing calibration means metric warnings are unavailable. Never say you "
    "are doing or have done something you were not given a tool for (opening apps, "
    "closing windows, calling people, changing settings); say you can't do that yet.")

SKILLS = ("describe", "read", "next_page", "find", "seat", "page_question", "chat")

ROUTER_INSTRUCTIONS = (
    "Decide what the user wants and reply with ONLY one JSON object on one line:\n"
    '{"skill": "<skill>", "name": "<object name or empty>", "say": "<what ARGUS says now>"}\n'
    "Skills:\n"
    "- describe: anything about what is around or in front of them, what something "
    "looks like, colours, people present, whether a path is clear\n"
    "- read: read a book, page, label, sign, letter, screen or any text, or answer "
    "a question about printed text that has not been read yet (the title, the "
    "author, a price, a date)\n"
    "- next_page: they turned the page and want it read\n"
    "- find: locate a named object; put the object in name\n"
    "- seat: find a place to sit\n"
    "- page_question: a question about the text that was just read\n"
    "- chat: anything else - greetings, small talk, general knowledge, how to do "
    "something, what ARGUS can do. General discussion of books, authors, or "
    "writing a title is chat, not read. Only use read for actual camera text.\n"
    'For chat, "say" is the full answer (at most two short sentences). For every '
    'other skill, "say" is a brief acknowledgement of at most six words, such as '
    '"Sure, let me look." or "Okay, hold the book up."')

_EXIT = re.compile(r"\b(shut ?down|switch off|turn (yourself )?off|power off|"
                   r"close (the )?(first[- ]person )?(view|window|screen|demo)|exit argus|"
                   r"quit argus|goodbye argus)\b", re.I)
_FILLER = re.compile(r"^\s*(u+m+|u+h+|hmm+|ah+|er+|okay|ok|yeah|hm)[\s.,!?]*$", re.I)
_TEXT_QUESTION = re.compile(
    r"\b(book|title|author|writer|wrote|cover|page|label|letter|sign|menu|packet|"
    r"medicine|bottle|text|written|says?|expiry|price)\b", re.I)
_STOP = re.compile(r"^\s*(stop|cancel|quiet|be quiet|shut up|enough)\b", re.I)
_PAUSE = re.compile(r"^\s*(pause|wait|hold on)\b", re.I)
_CONTINUE = re.compile(r"^\s*(continue|resume|go on|carry on|keep going)\b", re.I)
_REPEAT = re.compile(r"^\s*(repeat|again|say (that|it) again|read (that|it) again|"
                     r"what did you say)\b", re.I)
_NEXT_PAGE = re.compile(r"\b(next page|turned the page|new page|another page|"
                        r"read the next)\b", re.I)
_DESCRIBE = re.compile(r"\b(what('?s| is)( there)? (in front|around|ahead)|"
                       r"what (do|can) you see|describe|look around|where am i|"
                       r"what am i (looking|facing)|who('?s| is) (here|there|in front))\b", re.I)


@dataclass
class Intent:
    skill: str
    name: str = ""
    say: str = ""
    source: str = "rule"      # "rule" | "gemma"


@dataclass
class Turn:
    user: str
    argus: str
    skill: str
    t: float = field(default_factory=time.monotonic)


def rule_route(text: str, reading_active: bool, reading_playing: bool) -> Intent | None:
    """Deterministic routing for requests that must not depend on a model."""
    from .agent import requested_object
    from .reading import is_page_question, is_read_request
    from .seat import is_seat_request
    t = text.strip()
    if not t or _FILLER.search(t):
        return Intent("ignore")
    if _EXIT.search(t):
        return Intent("exit", say="Closing. Goodbye.")
    if _STOP.search(t):
        return Intent("control", "stop")
    if _PAUSE.search(t):
        return Intent("control", "pause")
    if _CONTINUE.search(t):
        return Intent("control", "continue")
    if _REPEAT.search(t):
        return Intent("control", "repeat")
    if _NEXT_PAGE.search(t):
        return Intent("next_page", say="Okay. Hold the next page up.")
    generic_read = is_read_request(t) or re.search(
        r"\b(help me|need to|want to|have to|can you|could you|would you|please)\s+"
        r"(?:\w+\s+){0,2}read\b", t, re.I)
    # Discussing text is not a request to capture text. Preserve explicit
    # reading requests and leave general discussion to the contextual router.
    conceptual_text = not generic_read and bool(re.search(
        r"\b(difference between|compare|recommend|suggest|come up with|"
        r"think of|write|create|invent)\b", t, re.I))
    if generic_read and not re.search(r"\b(title|author|writer|name of)\b", t, re.I):
        return Intent("read", say="Sure. Hold it up in front of you, about arm's length away.")
    if not conceptual_text and reading_active and is_page_question(t):
        return Intent("page_question")
    if not conceptual_text and (generic_read or re.search(r"\b(title|author|writer)\b", t, re.I) or (
            _TEXT_QUESTION.search(t) and re.search(
                r"\b(what|who|which|when|how much|tell me|read)\b", t, re.I))):
        # "What is the name of the book?", "who wrote this?", "read the title
        # and the author": read the text, then answer only that. Chat answered
        # "I cannot find the book title" without looking (2026-10-21).
        return Intent("read", name=t, say="Let me read it.")
    if is_seat_request(t):
        return Intent("seat", say="Okay, let me find you a seat.")
    name = requested_object(t)
    if name and not is_seat_request(t):
        return Intent("find", name=name, say="")
    if _DESCRIBE.search(t):
        return Intent("describe", say="")
    return None


_JSON = re.compile(r"\{.*\}", re.S)


def parse_router(text: str) -> Intent | None:
    """Validate Gemma's routing JSON; None if unusable."""
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    skill = str(obj.get("skill", "")).strip().lower()
    if skill not in SKILLS:
        return None
    name = str(obj.get("name") or "").strip()
    say = re.sub(r"\s+", " ", str(obj.get("say") or "")).strip()
    if skill == "find" and not name:
        skill = "describe"
    if skill == "chat" and not say:
        return None
    return Intent(skill, name, say[:300], "gemma")


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class DialogueManager:
    """One conversation. `handle()` is called with each transcribed or typed
    utterance; skills run through the orchestrator."""

    def __init__(self, orch, history_turns: int = 4):
        self.o = orch
        self.history: collections.deque[Turn] = collections.deque(maxlen=history_turns)
        from .companion import SessionMemory
        self.memory = getattr(orch, 'memory', None) or SessionMemory()
        self._reply_deadline = 0.0
        self.expect_reply = False      # bounded conversational follow-up window
        self.page_no = 0

    @property
    def expect_reply(self):
        return time.monotonic() < self._reply_deadline

    @expect_reply.setter
    def expect_reply(self, enabled):
        self._reply_deadline = time.monotonic()+20 if enabled else 0.0

    def _cancelled(self):
        tasks = getattr(self.o, 'tasks', None)
        return bool(tasks and tasks.cancelled())

    # ------------------------------------------------------------ context
    def _context_lines(self) -> str:
        lines = []
        depth = getattr(self.o,'depth',None)
        if depth is not None:
            lines.append('(Runtime: stereo calibration '+('loaded' if depth.calibrated else
                         'MISSING; metric warnings and navigation unavailable')+
                         '; wide-camera object distances are unverified.)')
        for t in self.history:
            lines.append(f"User: {t.user}\nARGUS: {t.argus}")
        r = getattr(self.o, "_reader", None)
        if r is not None and r.active:
            lines.append(f"(Active task: reading page {self.page_no}; the page text is "
                         "available for questions.)")
        observations = self.memory.context()
        if observations:
            lines.append("Tool observations (past views, not current safety):\n"+observations)
        from .sensitive import mask_text
        return mask_text("\n".join(lines))

    def _say(self, text: str) -> None:
        from .speech import Priority
        for s in sentences(text):
            if self._cancelled():
                break
            self.o.speaker.speak(s, Priority.NORMAL)

    # ------------------------------------------------------------ routing
    def route(self, text: str) -> Intent:
        from .companion import companion_route
        from .wake_stt import match_wake
        wake = match_wake(text)
        if wake.woke and wake.request:
            text = wake.request
        companion = companion_route(text, self.memory)
        if companion is not None:
            return Intent(companion[0], companion[1], companion[2])
        r = getattr(self.o, "_reader", None)
        intent = rule_route(text, bool(r and r.active), bool(r and r._thread
                                                             and r._thread.is_alive()))
        if intent is not None:
            return intent
        return self.gemma_route(text)

    def gemma_route(self, text: str) -> Intent:
        from .sensitive import mask_text
        text = mask_text(text)
        ctx = self._context_lines()
        messages = [
            {"role": "system", "content": PERSONA + "\n\n" + ROUTER_INSTRUCTIONS},
            {"role": "user", "content": (f"Conversation so far:\n{ctx}\n\n" if ctx else "")
             + f"User now says: {text}"},
        ]
        try:
            t0 = time.perf_counter()
            data = self.o.agent._post_text(messages, max_tokens=128)
            bus.set(router_latency=f"{time.perf_counter() - t0:.1f}s")
            intent = parse_router(data)
        except Exception as e:  # noqa: BLE001
            print(f"[dialogue] router failed: {e}")
            intent = None
        if intent is None:
            return Intent("answer", say="Could you say that another way? You can ask me to talk, find an object, or read a page.", source="fallback")
        return intent

    # ------------------------------------------------------------ handle
    def handle(self, text: str) -> None:
        intent = self.route(text)
        bus.record("intent", text=text, skill=intent.skill, name=intent.name,
                   source=intent.source)
        bus.set(intent=f"{intent.skill} ({intent.source})")
        reply = self.run(intent, text)
        if self._cancelled():
            return
        if intent.skill not in ("control", "ignore", "exit", "forget"):
            from .sensitive import mask_text
            self.history.append(Turn(mask_text(text)[:600], mask_text(reply or '')[:900], intent.skill))
            self.expect_reply = True

    def run(self, intent: Intent, text: str) -> str:
        o = self.o
        self.expect_reply = False
        if self._cancelled():
            return ''
        if intent.skill == 'preference_name':
            self.memory.preferences['name'] = intent.name
            answer = f'I will call you {intent.name} for this session.'
            self._say(answer)
            return answer
        if intent.skill == 'recap':
            turns = list(self.history)[-3:]
            answer = ('Recent requests: '+ '; '.join(t.user for t in turns)+'.'
                      if turns else 'We have no completed conversation turns in this session yet.')
            self._say(answer)
            return answer
        if intent.skill == 'preference':
            self.memory.preferences['detail'] = intent.name
            self._say(intent.say)
            return intent.say
        if intent.skill == 'reminder':
            seconds,message = json.loads(intent.name)
            reminders = getattr(o,'reminders',None)
            answer = ('Reminder set for this session.' if reminders and reminders.add(seconds,message)
                      else 'I could not set that reminder.')
            self._say(answer)
            return answer
        if intent.skill == 'clear_reminders':
            reminders = getattr(o,'reminders',None)
            if reminders:
                reminders.clear()
            self._say('Session timers and reminders cleared.')
            return 'Session timers and reminders cleared.'
        if intent.skill == 'answer':
            self._say(intent.say)
            return intent.say
        if intent.skill == 'forget':
            self.memory.forget()
            if getattr(o,'reminders',None):
                o.reminders.clear()
            self.history.clear()
            mapping = getattr(o,'mapping',None)
            if mapping is not None:
                mapping.map.reset(mapping.map.epoch)
            self._say('I cleared this session conversation, object observations and marked places. Existing study logs are unchanged.')
            return ''
        if intent.skill in ('inventory','watch','mark_place','places','navigate','location','status','safety_status'):
            from .assistance import run
            answer = run(o,intent.skill,intent.name)
            self._say(answer)
            return answer
        if intent.skill == "ignore":
            return ""
        if intent.skill == "exit":
            self._say(intent.say)
            o.speaker.wait_until_idle(timeout=5.0)
            o.request_exit()
            return intent.say
        if intent.skill == "control":
            if intent.name == "stop":
                o.seat_cancel.set()
                tasks = getattr(o,'tasks',None)
                if tasks is not None:
                    tasks.cancel()
            r = getattr(o, "_reader", None)
            if r is not None:
                r.control(intent.name)
            elif intent.name in ("stop", "pause"):
                o.speaker.cancel_normal()
            elif intent.name == "repeat" and self.history:
                self._say(self.history[-1].argus)
            return ""
        if intent.skill in ("read", "next_page"):
            if intent.say:
                self._say(intent.say)
            question = intent.name if intent.skill == "read" else ""
            reading = o.reader.read(intro=None, play=not question,
                                    on_finished=self._page_finished)
            if reading is None:
                return "I couldn't read it."
            if question:
                # Answer just what was asked, from the OCR text (masked).
                answer = o.agent.ask_text(question, o.reader.page_context(), max_tokens=64)
                self._say(answer or "I couldn't find that in the text.")
                self.expect_reply = True
                self.page_no = 1
                return answer
            self.page_no = self.page_no + 1 if intent.skill == "next_page" else 1
            return f"(read page {self.page_no}: {len(reading.lines)} lines)"
        if intent.skill == "seat":
            if intent.say:
                self._say(intent.say)
            from .seat import SeatSession
            ok = SeatSession(o).run()
            return "Guided to a seat." if ok else "Could not find a seat."
        if intent.skill == "find":
            answer = o._handle_query(f"find the {intent.name}")
            return answer or f"(looked for {intent.name}; outcome unavailable)"
        if intent.skill == "describe":
            if intent.say:
                self._say(intent.say)
            return o._handle_query(text) or "(scene response unavailable)"
        if intent.skill == "page_question":
            r = getattr(o, "_reader", None)
            if r is None or not r.active:
                return self.chat(text)
            answer = o.agent.ask_text(text, r.page_context(), max_tokens=64)
            self._say(answer or "I couldn't find that on the page.")
            return answer
        # chat
        if intent.say:
            self._say(intent.say)
            return intent.say
        return self.chat(text)

    def chat(self, text: str) -> str:
        from .sensitive import mask_text
        text = mask_text(text)
        ctx = self._context_lines()
        messages = [{"role": "system", "content": PERSONA},
                    {"role": "user", "content": (f"Conversation so far:\n{ctx}\n\n" if ctx else "")
                     + text}]
        answer = (self.o.agent._post_text(messages, max_tokens=120 if self.memory.preferences.get("detail") == "detail" else 80) or "").strip()
        self._say(answer or "Sorry, I didn't catch that.")
        return answer

    def _page_finished(self) -> None:
        self.expect_reply = True

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
    "worn by a blind person, in the spirit of a trusted butler. Speak naturally, "
    "in at most two short sentences. Use plain words, no lists, no markdown, no "
    "emoji. Confirm what you are doing. Never pretend to see something you have "
    "not been shown; never guess private numbers; for anything about safety, say "
    "you will watch and that the safety system warns automatically. Never say you "
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
    "something, what ARGUS can do\n"
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
    if generic_read and not re.search(r"\b(title|author|writer|name of)\b", t, re.I):
        return Intent("read", say="Sure. Hold it up in front of you, about arm's length away.")
    if reading_active and is_page_question(t):
        return Intent("page_question")
    if generic_read or re.search(r"\b(title|author|writer)\b", t, re.I) or (
            _TEXT_QUESTION.search(t) and re.search(
                r"\b(what|who|which|when|how much|tell me|read)\b", t, re.I)):
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
        self.expect_reply = False      # ARGUS asked something: listen without wake word
        self.page_no = 0

    # ------------------------------------------------------------ context
    def _context_lines(self) -> str:
        lines = []
        for t in self.history:
            lines.append(f"User: {t.user}\nARGUS: {t.argus}")
        r = getattr(self.o, "_reader", None)
        if r is not None and r.active:
            lines.append(f"(Active task: reading page {self.page_no}; the page text is "
                         "available for questions.)")
        return "\n".join(lines)

    def _say(self, text: str) -> None:
        from .speech import Priority
        for s in sentences(text):
            self.o.speaker.speak(s, Priority.NORMAL)

    # ------------------------------------------------------------ routing
    def route(self, text: str) -> Intent:
        r = getattr(self.o, "_reader", None)
        intent = rule_route(text, bool(r and r.active), bool(r and r._thread
                                                             and r._thread.is_alive()))
        if intent is not None:
            return intent
        return self.gemma_route(text)

    def gemma_route(self, text: str) -> Intent:
        ctx = self._context_lines()
        messages = [
            {"role": "system", "content": PERSONA + "\n\n" + ROUTER_INSTRUCTIONS},
            {"role": "user", "content": (f"Conversation so far:\n{ctx}\n\n" if ctx else "")
             + f"User now says: {text}"},
        ]
        try:
            t0 = time.perf_counter()
            data = self.o.agent._post_text(messages, max_tokens=90)
            bus.set(router_latency=f"{time.perf_counter() - t0:.1f}s")
            intent = parse_router(data)
        except Exception as e:  # noqa: BLE001
            print(f"[dialogue] router failed: {e}")
            intent = None
        if intent is None:
            # Unusable routing: a visual question is the safest useful default.
            return Intent("describe", say="", source="fallback")
        return intent

    # ------------------------------------------------------------ handle
    def handle(self, text: str) -> None:
        intent = self.route(text)
        bus.record("intent", text=text, skill=intent.skill, name=intent.name,
                   source=intent.source)
        bus.set(intent=f"{intent.skill} ({intent.source})")
        reply = self.run(intent, text)
        if intent.skill not in ("control", "ignore", "exit"):
            self.history.append(Turn(text, reply, intent.skill))

    def run(self, intent: Intent, text: str) -> str:
        o = self.o
        self.expect_reply = False
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
            o._handle_query(f"find the {intent.name}")
            return f"(looked for {intent.name})"
        if intent.skill == "describe":
            if intent.say:
                self._say(intent.say)
            o._handle_query(text)
            return "(described the scene)"
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
        ctx = self._context_lines()
        messages = [{"role": "system", "content": PERSONA},
                    {"role": "user", "content": (f"Conversation so far:\n{ctx}\n\n" if ctx else "")
                     + text}]
        answer = (self.o.agent._post_text(messages, max_tokens=80) or "").strip()
        self._say(answer or "Sorry, I didn't catch that.")
        return answer

    def _page_finished(self) -> None:
        from .speech import Priority
        self.o.speaker.speak("That's the end of the page. Say next page when you've "
                             "turned it, or ask me about it.", Priority.NORMAL)
        self.expect_reply = True

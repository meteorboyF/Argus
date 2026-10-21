"""Conversation layer: deterministic routing, Gemma routing validation, and
the book-reading conversation with fake skills."""
from __future__ import annotations

import threading

import pytest

from argus.dialogue import DialogueManager, Intent, parse_router, rule_route, sentences


@pytest.mark.parametrize("text,skill,name", [
    ("stop", "control", "stop"), ("pause please", "control", "pause"),
    ("go on", "control", "continue"), ("read that again", "control", "repeat"),
    ("next page", "next_page", ""), ("I turned the page", "next_page", ""),
    ("Argus, help me read this book", "read", ""), ("read this", "read", ""),
    ("I need to read my medicine box", "read", ""), ("can you please read the label", "read", ""),
    ("find me a seat", "seat", ""), ("where are my keys", "find", "keys"),
    ("what's in front of me", "describe", ""), ("describe the room", "describe", ""),
])
def test_rule_route(text, skill, name):
    i = rule_route(text, reading_active=False, reading_playing=False)
    assert i is not None and i.skill == skill and i.name == name


def test_text_question_reads_first_unless_a_page_is_active():
    # Nothing read yet: read the label, then answer just that question.
    i = rule_route("what's the expiry date?", False, False)
    assert i.skill == "read" and i.name == "what's the expiry date?"
    # A page is active: answer from it without re-reading.
    assert rule_route("what's the expiry date?", True, False).skill == "page_question"


def test_small_talk_goes_to_gemma():
    assert rule_route("how are you today?", False, False) is None
    assert rule_route("what can you do?", False, False) is None


@pytest.mark.parametrize("raw,expect", [
    ('{"skill": "chat", "name": "", "say": "I am well, thank you."}', ("chat", "I am well, thank you.")),
    ('Sure! {"skill":"find","name":"remote","say":"Let me look."}', ("find", "Let me look.")),
    ('{"skill": "find", "name": "", "say": "Okay."}', ("describe", "Okay.")),   # no name
    ('{"skill": "launch_missiles", "say": "ok"}', None),                     # not a skill
    ('{"skill": "chat", "say": ""}', None),                                  # empty chat
    ("I think you want to read.", None),                                    # no JSON
])
def test_parse_router(raw, expect):
    i = parse_router(raw)
    assert (None if i is None else (i.skill, i.say)) == expect


def test_sentences():
    assert sentences("Sure. Hold it up! Ready?") == ["Sure.", "Hold it up!", "Ready?"]


# ---------------------------------------------------------------- fakes
class FakeSpeaker:
    def __init__(self):
        self.said = []
        self.cancelled = 0

    def speak(self, text, priority=None):
        self.said.append(text)

    def cancel_normal(self):
        self.cancelled += 1


class FakeReading:
    def __init__(self):
        self.lines = [1, 2, 3]


class FakeReader:
    def __init__(self):
        self.reads, self.controls = 0, []
        self.active = False
        self._thread = None

    def read(self, coach=True, intro=None, on_finished=None, play=True):
        self.reads += 1
        self.active = True
        on_finished()                         # the whole page was spoken
        return FakeReading()

    def control(self, w):
        self.controls.append(w)

    def page_context(self):
        return "Paracetamol 500 mg. Take one tablet every 6 hours. EXP 03/2027."


class FakeAgent:
    def __init__(self, router_reply="", chat_reply=""):
        self.router_reply, self.chat_reply = router_reply, chat_reply
        self.texts = []

    def _post_text(self, messages, max_tokens=80, temperature=0.2):
        self.texts.append(messages[-1]["content"])
        return self.router_reply if "JSON" in messages[0]["content"] else self.chat_reply

    def ask_text(self, q, ctx, max_tokens=None):
        return "It expires in March 2027." if "expir" in q else "Not on the page."


class FakeOrch:
    def __init__(self, agent):
        self.speaker = FakeSpeaker()
        self.agent = agent
        self.reader = FakeReader()
        self._reader = self.reader
        self.seat_cancel = threading.Event()
        self.queries = []

    def _handle_query(self, q):
        self.queries.append(q)


def test_book_reading_conversation():
    o = FakeOrch(FakeAgent())
    d = DialogueManager(o)
    d.handle("Argus, help me read this book")
    assert o.reader.reads == 1 and d.page_no == 1
    assert o.speaker.said[0].startswith("Sure.")
    assert "Say next page" in o.speaker.said[-1] and d.expect_reply
    d.handle("what's the expiry date?")
    assert o.speaker.said[-1] == "It expires in March 2027."
    d.handle("next page")
    assert o.reader.reads == 2 and d.page_no == 2
    d.handle("stop")
    assert o.reader.controls[-1] == "stop" and o.seat_cancel.is_set()


def test_small_talk_is_answered_by_gemma_with_history():
    agent = FakeAgent(router_reply='{"skill":"chat","name":"","say":"I am well, thanks. How can I help?"}')
    o = FakeOrch(agent)
    o._reader = None
    d = DialogueManager(o)
    d.handle("how are you?")
    assert o.speaker.said == ["I am well, thanks.", "How can I help?"]
    d.handle("tell me a joke")
    assert "User: how are you?" in agent.texts[-1]          # history reaches the model


def test_gemma_routes_a_visual_question_to_the_camera():
    agent = FakeAgent(router_reply='{"skill":"describe","name":"","say":"Let me look."}')
    o = FakeOrch(agent)
    o._reader = None
    DialogueManager(o).handle("is the table clear?")
    assert o.queries == ["is the table clear?"] and o.speaker.said == ["Let me look."]


def test_unusable_router_output_asks_for_clarification():
    o = FakeOrch(FakeAgent(router_reply="hmm"))
    o._reader = None
    DialogueManager(o).handle("anything interesting nearby")
    assert o.queries == []
    assert o.speaker.said[0] == "Could you say that another way?"


def test_find_uses_grounding_path():
    o = FakeOrch(FakeAgent())
    DialogueManager(o).handle("where are my keys?")
    assert o.queries == ["find the keys"]


@pytest.mark.parametrize("text", ["shut down", "close the first person view", "exit argus",
                                  "Jarvis, turn yourself off"])
def test_exit_commands(text):
    assert rule_route(text, False, False).skill == "exit"


@pytest.mark.parametrize("text", ["Um...", "uh", "hmm", "okay", ""])
def test_filler_is_ignored(text):
    assert rule_route(text, False, False).skill == "ignore"


def test_exit_speaks_and_requests_exit():
    o = FakeOrch(FakeAgent())
    o.speaker.wait_until_idle = lambda timeout=None: True
    o.exited = False
    o.request_exit = lambda: setattr(o, "exited", True)
    DialogueManager(o).handle("shut down")
    assert o.speaker.said == ["Closing.", "Goodbye."] and o.exited


@pytest.mark.parametrize("text", ["What is the name of the book?", "who wrote this book",
                                  "what does the label say about the dose",
                                  "read the title of the book and the name of the writer",
                                  "I'm the title of the book and the name of the writer"])
def test_questions_about_printed_text_go_to_reading(text):
    i = rule_route(text, False, False)
    assert i.skill == "read"


def test_title_question_answers_only_that():
    o = FakeOrch(FakeAgent())
    calls = {}

    def read(coach=True, intro=None, on_finished=None, play=True):
        calls["play"] = play
        o.reader.active = True
        return FakeReading()
    o.reader.read = read
    o.agent.ask_text = lambda q, ctx, max_tokens=None: "The book is called The Little Prince."
    DialogueManager(o).handle("What is the name of the book?")
    assert calls["play"] is False
    assert o.speaker.said == ["Let me read it.", "The book is called The Little Prince."]

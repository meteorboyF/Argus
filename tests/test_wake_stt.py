"""Speech-triggered 'Argus' wake word matching."""
from __future__ import annotations

import pytest

from argus.wake_stt import match_wake


@pytest.mark.parametrize("text,expected", [
    ("Hey Argus.", ""),
    ("Argus", ""),
    ("argus?", ""),
    ("Hey Argus, what's in front of me?", "what's in front of me"),
    ("Argus, help me read this book.", "help me read this book"),
    ("Hey, Argos, find me a seat", "find me a seat"),
    ("Hi August, what can you see?", "what can you see"),
    ("Argus stop", "stop"),
    ("Okay Argus, read this", "read this"),
])
def test_wakes(text, expected):
    m = match_wake(text)
    assert m.woke and m.request == expected


@pytest.mark.parametrize("text", [
    "I was reading about Argus yesterday",       # name not at the start
    "What time is it?",
    "Hey Jarvis",
    "Arguments are fun",                          # different word
    "",
])
def test_does_not_wake(text):
    assert not match_wake(text).woke

"""Speech-triggered wake word: "Hey Argus" / "Argus" without a trained model.

A custom openWakeWord "Hey Argus" model needs hours of off-device GPU
training (training/hey_argus/). Until it exists, ARGUS wakes on its own name
by transcribing short speech segments:

  Silero VAD cuts each spoken phrase (0.3-8 s)
  -> faster-whisper tiny (CPU), biased toward the word "Argus"
  -> if the phrase starts with "Argus" / "Hey Argus" (or a common mishearing)
     the rest of the phrase IS the request ("Argus, what's in front of me?");
     a bare "Argus" gets the listening earcon and the next phrase.

Trade-off, measured and logged: Whisper runs on every nearby phrase (CPU),
and "Argus" said inside an unrelated sentence can wake it. Only the start of
a phrase counts, which keeps that rare.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# "argus" and what tiny Whisper tends to hear for it.
_NAME = r"(?:argus|argos|argas|argues|arcus|aargus|are\s?gus|our\s?gus|r\s?gus|august|augus)"
_WAKE = re.compile(r"^\W*(?:(?:hey|hi|hello|okay|ok|oi|yo)\W+)?" + _NAME + r"\b[\s,.!?:;-]*",
                   re.IGNORECASE)

# Measured 2026-10-21 on Piper-synthesised phrases: a prefix-style hint
# ("Argus, hey Argus.") made Whisper drop the name from "Argus, what's in
# front of me?"; a glossary-style hint scored 9/9 (wakes on every Argus
# phrase, not on "Hey guys, come here" or "Arguments about art") at 0.78 s.
WHISPER_HINT = "Names: Argus."


@dataclass
class WakeMatch:
    woke: bool
    request: str = ""          # text after the name ("" = bare wake word)


def match_wake(transcript: str) -> WakeMatch:
    t = (transcript or "").strip()
    m = _WAKE.match(t)
    if not m:
        return WakeMatch(False)
    rest = t[m.end():].strip(" ,.!?;:-")
    # "Argus." alone, or followed by a filler, is a bare wake.
    if len(rest.split()) < 2 and not re.search(r"\b(stop|pause|repeat|continue|help)\b",
                                                rest, re.IGNORECASE):
        rest = ""
    return WakeMatch(True, rest)


def transcribe_hinted(stt, audio) -> str:
    """Transcribe with the name as a prompt so Whisper spells it 'Argus'."""
    segments, _ = stt.model.transcribe(audio, language="en", initial_prompt=WHISPER_HINT,
                                       beam_size=1, condition_on_previous_text=False)
    text = " ".join(s.text for s in segments).strip()
    # The prompt can leak into silence-only audio; never treat that as speech.
    return "" if text.lower().strip(" .,!:") in ("names: argus", "names") else text

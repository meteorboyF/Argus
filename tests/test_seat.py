"""Seat guidance decisions (M7)."""
from __future__ import annotations

import numpy as np
import pytest

from argus.seat import (SeatGuide, SeatSighting, bearing_deg, detect_seats, direction_words,
                        guidance_phrase, is_seat_request)

W, F = 960, 550.0


def _seat(cx, w=120, cls="chair", score=0.8):
    return SeatSighting(np.array([cx - w / 2, 300, cx + w / 2, 500], np.float32), score, cls)


@pytest.mark.parametrize("bearing,words", [(0, "straight ahead"), (-10, "slightly left"),
                                           (30, "to your right"), (-60, "far left")])
def test_direction_words(bearing, words):
    assert direction_words(bearing) == words


def test_phrases():
    assert guidance_phrase(12.0, 2.8) == "Seat, slightly right, 4 steps."
    assert guidance_phrase(-3.0, 0.7) == "Seat in front of you, knee height."
    assert guidance_phrase(-3.0, 1.0) == "Seat, straight ahead, 1 step."
    assert guidance_phrase(25.0, None) == "Seat, to your right."     # uncalibrated: no steps


def test_bearing_from_box():
    assert bearing_deg(np.array([430, 0, 530, 10]), W, F) == pytest.approx(0.0, abs=0.1)
    assert bearing_deg(np.array([700, 0, 800, 10]), W, F) > 25


def test_guide_prefers_nearest_seat_with_depth():
    d = np.full((600, W), 9.0, np.float32)
    d[300:500, 150:250] = 4.0             # left chair, 4 m
    d[300:500, 700:820] = 2.1             # right chair, 2.1 m
    g = SeatGuide(F)
    phrase = g.update([_seat(200), _seat(760)], W, d, 0.0)
    assert phrase.startswith("Seat, to your right, 3 steps")


def test_guide_without_depth_uses_most_central():
    g = SeatGuide(F)
    assert g.update([_seat(150), _seat(520)], W, None, 0.0) == "Seat, straight ahead."


def test_lost_target_turns_toward_last_side():
    g = SeatGuide(F, lost_after_s=1.0)
    g.update([_seat(800)], W, None, 0.0)
    assert g.update([], W, None, 0.5) == "Seat, to your right."   # still the last phrase
    assert g.update([], W, None, 1.6) == "Turn slowly right. I am looking for a seat."


def test_arrival():
    d = np.full((600, W), 0.7, np.float32)
    g = SeatGuide(F)
    g.update([_seat(480)], W, d, 0.0)
    assert g.state.arrived


def test_decode_seat_classes_only():
    raw = np.zeros((84, 100), np.float32)
    raw[:4, 0] = (320, 208, 80, 120)
    raw[4 + 56, 0] = 0.9                       # chair
    raw[:4, 1] = (100, 100, 50, 150)
    raw[4 + 0, 1] = 0.95                       # person: ignored
    seats = detect_seats(raw, 1.0, 0, 0, (416, 640))
    assert [s.cls for s in seats] == ["chair"]


@pytest.mark.parametrize("q,yes", [("find me a seat", True), ("where can I sit down", True),
                                   ("I want to sit", True), ("find a chair", True),
                                   ("read this", False), ("where is the door", False)])
def test_seat_requests(q, yes):
    assert is_seat_request(q) == yes

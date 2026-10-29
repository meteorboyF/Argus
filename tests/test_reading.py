"""Reading mode: OCR ordering, speech chunks, sensitive-text masking."""
from __future__ import annotations

import numpy as np
import pytest

from argus.ocr import Word, order_lines, speakable_chunks
from argus.sensitive import blur_boxes, find_sensitive, luhn_ok, mask_text


def _w(text, x, y, w=80, h=20, score=0.9):
    box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
    return Word(text, box, score)


def test_words_are_grouped_into_lines_in_reading_order():
    words = [_w("EXIT", 200, 100), _w("Room", 100, 200), _w("FIRE", 100, 102),
             _w("204", 200, 203), _w("closed", 300, 300), _w("Keep", 100, 298)]
    lines = order_lines(words)
    assert [l.text for l in lines] == ["FIRE EXIT", "Room 204", "Keep closed"]


def test_slightly_skewed_line_stays_together():
    words = [_w("left", 100, 100), _w("middle", 200, 106), _w("right", 300, 112)]
    assert [l.text for l in order_lines(words)] == ["left middle right"]


def test_speakable_chunks_split_sentences_and_keep_sign_lines():
    lines = order_lines([_w("FIRE EXIT", 100, 50),
                         _w("Take one tablet daily.", 100, 100),
                         _w("Do not exceed the dose. Keep", 100, 130),
                         _w("away from children.", 100, 160)])
    assert speakable_chunks(lines) == ["FIRE EXIT", "Take one tablet daily.",
                                       "Do not exceed the dose.", "Keep away from children."]


@pytest.mark.parametrize("num,ok", [("4111 1111 1111 1111", True), ("4111111111111112", False),
                                    ("5500-0000-0000-0004", True)])
def test_luhn(num, ok):
    assert luhn_ok(num) == ok


@pytest.mark.parametrize("text,kinds", [
    ("Card 4111 1111 1111 1111 exp 12/27", ["card"]),
    ("Call 01712345678 now", ["phone"]),
    ("Call +880 1812-345678", ["phone"]),
    ("Your OTP is 482913. Do not share.", ["code"]),
    ("আপনার ওটিপি 5521", ["code"]),
    ("Room 204, open 9 to 5", []),
    ("Order 123456 shipped", []),           # number without a keyword is not a code
    ("Invoice 4111111111111112", []),       # fails Luhn
])
def test_find_sensitive(text, kinds):
    assert [f.kind for f in find_sensitive(text)] == kinds


def test_mask_text():
    assert mask_text("OTP 4829 for card 4111 1111 1111 1111") == "OTP [hidden] for card [hidden]"


def test_blur_boxes_changes_only_the_box():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 300, 3), np.uint8)
    box = np.array([[50, 50], [150, 50], [150, 80], [50, 80]])
    out = blur_boxes(img, [box])
    assert not np.array_equal(out[50:80, 50:150], img[50:80, 50:150])
    assert np.array_equal(out[150:, :], img[150:, :])


# --------------------------------------------------------------------- intents
from argus.reading import (FrameCheck, control_word, framing_advice, is_page_question,  # noqa: E402
                           is_read_request)


@pytest.mark.parametrize("q", ["read this", "Read it for me", "please read the label",
                               "what does it say?", "What does the sign say"])
def test_read_requests(q):
    assert is_read_request(q)


@pytest.mark.parametrize("q", ["where are my keys", "what is in front of me", "ready"])
def test_not_read_requests(q):
    assert not is_read_request(q)


def test_controls_and_page_questions():
    assert control_word("stop") == "stop" and control_word("Resume please") == "continue"
    assert control_word("again") == "repeat" and control_word("read this") is None
    assert is_page_question("what's the expiry date?")
    assert not is_page_question("read the label")
    assert not is_page_question("where is the door")


# --------------------------------------------------------------------- framing
def _box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


@pytest.mark.parametrize("boxes,cue,ready", [
    ([], "I can't see the page yet", False),
    ([_box(0, 200, 500, 240)], "Move the page to your right.", False),
    ([_box(400, 200, 960, 240)], "Move the page to your left.", False),
    ([_box(200, 0, 700, 40)], "Move the page down", False),
    ([_box(420, 250, 520, 280)], "Bring the page a little closer.", False),
    ([_box(200, 150, 760, 190), _box(200, 220, 700, 260)], "The page is framed. Hold it there.", True),
])
def test_framing_advice(boxes, cue, ready):
    msg, ok = framing_advice(FrameCheck(boxes, 960, 540, sharpness=200.0, motion=1.0))
    assert msg.startswith(cue) and ok == ready


def test_motion_and_blur_block_capture():
    good = [_box(200, 150, 760, 190)]
    assert framing_advice(FrameCheck(good, 960, 540, 200.0, 20.0)) == ("Hold still.", False)
    small = [_box(300, 150, 600, 190)]
    msg, ok = framing_advice(FrameCheck(small, 960, 540, 10.0, 1.0))
    assert "blurry" in msg and not ok


def test_large_but_blurry_page_is_too_close_for_the_fixed_focus_lens():
    msg, ok = framing_advice(FrameCheck([_box(100, 100, 860, 400)], 960, 540, 10.0, 1.0))
    assert msg == ("I can see it, but it's blurry. Move it a bit back "
                   "so I can see it properly.") and not ok


def _page(**kw):
    base = dict(boxes=[], width=960, height=540, sharpness=200.0, motion=1.0,
                page_box=(300, 150, 650, 480), region_mean=120.0, background_mean=130.0,
                glare_frac=0.0)
    base.update(kw)
    return FrameCheck(**base)


def test_backlit_page_asks_to_turn_around():
    msg, ok = framing_advice(_page(region_mean=40.0, background_mean=190.0))
    assert "bright light behind the page" in msg and "Turn around" in msg and not ok


def test_dark_room_asks_for_light():
    msg, _ = framing_advice(_page(region_mean=30.0, background_mean=35.0))
    assert msg.startswith("It's too dark for me to read.")


def test_glare_asks_to_tilt():
    assert framing_advice(_page(glare_frac=0.1))[0] == "There's a reflection on the page. Tilt it a little."


def test_book_seen_without_text_never_captures():
    msg, ok = framing_advice(_page())
    assert not ok and msg.startswith("I can see the page")
    msg, ok = framing_advice(_page(sharpness=10.0, page_box=(100, 60, 860, 500)))
    assert msg.startswith("I can see it, but it's blurry. Move it a bit back") and not ok


def test_light_is_checked_before_framing():
    # Cut off at the left AND backlit: the light problem is said first.
    msg, _ = framing_advice(_page(page_box=(0, 150, 400, 480), region_mean=40.0,
                                  background_mean=200.0))
    assert "bright light behind the page" in msg


def test_cjk_noise_is_dropped_but_bangla_and_latin_kept():
    from argus.ocr import plausible
    assert not plausible("米") and not plausible("  ")
    assert plausible("EXIT") and plausible("৳ ১২০") and plausible("9:00AM-5:30PM")


def test_white_text_on_dark_cover_can_capture_but_dim_text_cannot():
    from argus.reading import text_contrast
    gray = np.full((540, 960), 20, np.uint8)
    box = _box(200, 150, 700, 250)
    # Broad white strokes on a dark cover, with a dark overall mean.
    for x in range(215, 690, 35):
        gray[165:235, x:x+9] = 210
    contrast = text_contrast(gray, [box])
    fc = _page(boxes=[box], region_mean=40, text_contrast=contrast)
    assert contrast >= 35 and framing_advice(fc)[1]
    dim = np.full_like(gray, 20)
    dim[gray == 210] = 27
    fc.text_contrast = text_contrast(dim, [box])
    assert not framing_advice(fc)[1]
    fc.text_contrast = contrast
    fc.motion = 10
    assert not framing_advice(fc)[1]
    fc.motion = 0
    fc.glare_frac = .1
    assert not framing_advice(fc)[1]

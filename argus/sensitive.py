"""Sensitive-text detection and masking for OCR output (privacy, M5).

Applied before any OCR text is logged, shown on the demo view, or sent to a
model, and before an image region containing it is displayed:

- payment card numbers (13-19 digits that pass the Luhn check),
- Bangladeshi mobile numbers, (+880)1[3-9]XXXXXXXX,
- one-time codes and PINs: a 4-8 digit number near OTP/PIN/code/password
  keywords (English and Bangla).

Speaking the text to the user who is holding the page is allowed: it is their
document. Masking protects everything that is stored or shown to others.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

_DIGITS_RUN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_BD_MOBILE = re.compile(r"(?<!\d)(?:\+?880[ -]?)?0?1[3-9]\d{2}[ -]?\d{6}(?!\d)")
_OTP_KEYWORDS = re.compile(
    r"\b(?:otp|pin|passcode|password|verification code|security code|one[- ]time|"
    r"cvv|cvc)\b|ওটিপি|পিন|পাসওয়ার্ড|গোপন",
    re.IGNORECASE)
_SHORT_CODE = re.compile(r"(?<!\d)\d{4,8}(?!\d)")

MASK = "[hidden]"


def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, v in enumerate(reversed(d)):
        if i % 2 == 1:
            v *= 2
            if v > 9:
                v -= 9
        total += v
    return total % 10 == 0


@dataclass
class Finding:
    kind: str       # "card" | "phone" | "code"
    start: int
    end: int


def find_sensitive(text: str) -> list[Finding]:
    out: list[Finding] = []
    for m in _DIGITS_RUN.finditer(text):
        if luhn_ok(m.group()):
            out.append(Finding("card", m.start(), m.end()))
    for m in _BD_MOBILE.finditer(text):
        if not any(f.start <= m.start() < f.end for f in out):
            out.append(Finding("phone", m.start(), m.end()))
    if _OTP_KEYWORDS.search(text):
        for m in _SHORT_CODE.finditer(text):
            if not any(f.start <= m.start() < f.end for f in out):
                out.append(Finding("code", m.start(), m.end()))
    return sorted(out, key=lambda f: f.start)


def mask_text(text: str) -> str:
    for f in reversed(find_sensitive(text)):
        text = text[:f.start] + MASK + text[f.end:]
    return text


def is_sensitive(text: str) -> bool:
    return bool(find_sensitive(text))


def blur_boxes(img: np.ndarray, boxes, kernel: int = 51) -> np.ndarray:
    """Blur quadrilateral text boxes (4x2 points each) in a copy of img."""
    out = img.copy()
    k = kernel | 1
    for box in boxes:
        pts = np.asarray(box, np.int32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts)
        x, y = max(0, x - 4), max(0, y - 4)
        roi = out[y:y + h + 8, x:x + w + 8]
        if roi.size:
            out[y:y + h + 8, x:x + w + 8] = cv2.GaussianBlur(roi, (k, k), 0)
    return out

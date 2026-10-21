"""Verbatim text reading (M5).

Small VLMs give confident misreads on text, so ARGUS reads text with a
dedicated OCR engine (PP-OCR det+rec via RapidOCR / ONNX Runtime, CPU) and
uses Gemma only for questions *about* the recognised text, as text.

Measured 2026-10-21: the synthetic sign reads correctly in ~1.2 s warm on the
Orin CPU (detection ~0.9 s of that at 1920x1080).

Reading order: OCR returns word/line boxes in arbitrary order. Boxes are
clustered into lines by vertical overlap, lines sorted top to bottom, words
left to right, then lines are grouped into blocks by vertical gaps so the
text can be spoken sentence by sentence.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .sensitive import find_sensitive, mask_text


@dataclass
class Word:
    text: str
    box: np.ndarray          # 4x2 points, image coordinates
    score: float

    @property
    def x0(self) -> float:
        return float(self.box[:, 0].min())

    @property
    def y0(self) -> float:
        return float(self.box[:, 1].min())

    @property
    def y1(self) -> float:
        return float(self.box[:, 1].max())

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class Line:
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.x0)).strip()

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def score(self) -> float:
        return float(np.mean([w.score for w in self.words]))


@dataclass
class Reading:
    lines: list[Line]
    elapsed_s: float
    sensitive: bool

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def masked_text(self) -> str:
        return mask_text(self.text)

    def sensitive_boxes(self) -> list[np.ndarray]:
        return [w.box for l in self.lines for w in l.words if find_sensitive(w.text)
                or find_sensitive(l.text)]


def order_lines(words: list[Word], overlap: float = 0.5) -> list[Line]:
    """Cluster words into lines: a word joins a line when their vertical
    extents overlap by at least `overlap` of the smaller height."""
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: w.cy):
        best, best_ov = None, 0.0
        for line in lines:
            inter = min(w.y1, line.y1) - max(w.y0, line.y0)
            h = min(w.height, line.y1 - line.y0) or 1.0
            ov = inter / h
            if ov > best_ov:
                best, best_ov = line, ov
        if best is not None and best_ov >= overlap:
            best.words.append(w)
        else:
            lines.append(Line([w]))
    lines.sort(key=lambda l: l.y0)
    return lines


_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def speakable_chunks(lines: list[Line], max_chars: int = 160) -> list[str]:
    """Sentences for streamed speech. Lines without final punctuation (signs,
    labels, headings) are spoken as their own chunk."""
    chunks: list[str] = []
    buf = ""
    for line in lines:
        t = line.text
        if not t:
            continue
        if not buf and not re.search(r"[.!?,;:।]$", t) and len(t) < 60:
            chunks.append(t)
            continue
        buf = f"{buf} {t}".strip()
        parts = _SENTENCE_END.split(buf)
        chunks.extend(p for p in parts[:-1] if p)
        buf = parts[-1]
        if len(buf) > max_chars:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


_CJK = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]")


def plausible(text: str) -> bool:
    """RapidOCR's default recogniser is a Chinese+English model; on a scene
    with no text it reported a lone '米' (2026-10-21). ARGUS reads English
    (and later Bangla), so tokens that are mostly CJK are noise."""
    t = text.strip()
    if not t:
        return False
    cjk = len(_CJK.findall(t))
    return cjk == 0 or cjk / len(t) < 0.5


class OCREngine:
    """Fail-closed wrapper: raises instead of returning partial garbage."""

    def __init__(self, min_score: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()
        self.min_score = min_score

    def read(self, img_bgr: np.ndarray) -> Reading:
        t0 = time.perf_counter()
        result, _ = self._ocr(img_bgr)
        words = []
        for box, text, score in result or []:
            score = float(score)
            if score >= self.min_score and plausible(str(text)):
                words.append(Word(str(text).strip(), np.asarray(box, np.float32), score))
        lines = order_lines(words)
        text = "\n".join(l.text for l in lines)
        return Reading(lines, time.perf_counter() - t0, bool(find_sensitive(text)))

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


def enhance(img_bgr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Local contrast equalisation on lightness only (LAB), colour kept.

    Light lettering on a dark book cover was invisible to the detector: no
    text at all on the raw still, 'Crime and Punishment' (0.86) once the
    crop was CLAHE-equalised (2026-10-21)."""
    import cv2
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clip, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_box(img: np.ndarray, box, pad: float = 0.06, min_side: int = 1400) -> np.ndarray:
    """Padded crop, upscaled so its longer side is at least `min_side`."""
    import cv2
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = img.shape[:2]
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    c = img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
    s = min_side / max(c.shape[:2]) if c.size else 1.0
    if s > 1.0:
        c = cv2.resize(c, None, fx=min(s, 3.0), fy=min(s, 3.0), interpolation=cv2.INTER_CUBIC)
    return c


class OCREngine:
    """Fail-closed wrapper: raises instead of returning partial garbage."""

    def __init__(self, min_score: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()
        self.min_score = min_score

    def read_robust(self, img_bgr: np.ndarray, locate=None,
                    objects=("book", "paper", "label", "sign")) -> Reading:
        """Locate a document first (`locate(name, img) -> Detection | None`,
        the YOLO-World grounder, ~60 ms) and read its contrast-enhanced,
        upscaled crop (~1 s); otherwise, or if that finds little, read the
        enhanced full still. Full-still-first took 4.9 s for a book cover."""
        import cv2
        t0 = time.perf_counter()
        best: Reading | None = None
        if locate is not None:
            small = cv2.resize(img_bgr, (1920, int(img_bgr.shape[0] * 1920 / img_bgr.shape[1])))
            k = img_bgr.shape[1] / 1920.0
            for name in objects:
                det = locate(name, small)
                if det is None:
                    continue
                r = self.read(enhance(crop_box(img_bgr, [v * k for v in det.bbox])))
                if best is None or len(r.text) > len(best.text):
                    best = r
                if len(best.lines) >= 1:
                    break
        if best is None or not best.lines:
            full = self.read(enhance(img_bgr))
            if best is None or len(full.text) > len(best.text):
                best = full
        best.elapsed_s = time.perf_counter() - t0
        return best

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

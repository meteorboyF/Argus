"""Precomputed CLIP text embeddings for the grounding vocabulary.

Loading torch + CLIP to embed one label costs ~1.4 GB of resident memory and
~10 s of cold start on the Orin Nano. The labels ARGUS is asked to find are
mostly predictable, so they are embedded once offline
(`scripts/build_vocab_embeddings.py`) and looked up here. Only a label that is
not in the table falls back to the live CLIP text encoder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_ARTICLE = re.compile(r"^(?:my|the|a|an|some)\s+", re.IGNORECASE)


def normalise(name: str) -> str:
    """Canonical lookup key: lower case, single spaces, no leading article."""
    s = re.sub(r"\s+", " ", str(name).strip().lower())
    s = s.strip(" .?!,")
    return _ARTICLE.sub("", s)


def _singular(s: str) -> str | None:
    if s.endswith("ies") and len(s) > 4:
        return s[:-3] + "y"
    if s.endswith("es") and s[:-2].endswith(("sh", "ch", "x", "s")):
        return s[:-2]
    if s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        return s[:-1]
    return None


class VocabTable:
    """Read-only label -> (1, 1, 512) float32 normalised embedding."""

    def __init__(self, words: list[str], vectors: np.ndarray, meta: dict | None = None):
        if vectors.shape != (len(words), 512):
            raise ValueError(f"vocab shape {vectors.shape} does not match {len(words)} words")
        self.meta = meta or {}
        self._index = {normalise(w): i for i, w in enumerate(words)}
        self._vectors = vectors.astype(np.float32)

    @classmethod
    def load(cls, npy_path: str | Path) -> "VocabTable":
        npy_path = Path(npy_path)
        meta = json.loads(npy_path.with_suffix(".json").read_text())
        return cls(meta["words"], np.load(npy_path), meta)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def get(self, name: str) -> np.ndarray | None:
        key = normalise(name)
        i = self._index.get(key)
        if i is None and (sing := _singular(key)) is not None:
            i = self._index.get(sing)
        if i is None:
            return None
        return self._vectors[i].reshape(1, 1, 512)

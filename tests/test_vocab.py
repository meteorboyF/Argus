"""Precomputed grounding vocabulary lookup."""
from __future__ import annotations

import numpy as np
import pytest

from argus.vocab import VocabTable, normalise


def _table():
    words = ["chair", "keys", "battery", "glass door", "cng auto rickshaw"]
    rng = np.random.default_rng(0)
    v = rng.normal(size=(len(words), 512)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return VocabTable(words, v), v


@pytest.mark.parametrize("raw,key", [("  The Chair? ", "chair"), ("my keys", "keys"),
                                     ("a  Glass   Door", "glass door")])
def test_normalise(raw, key):
    assert normalise(raw) == key


def test_lookup_shapes_and_values():
    t, v = _table()
    got = t.get("my chair")
    assert got.shape == (1, 1, 512) and got.dtype == np.float32
    assert np.allclose(got[0, 0], v[0])


def test_plural_falls_back_to_singular():
    t, v = _table()
    assert np.allclose(t.get("chairs")[0, 0], v[0])
    assert np.allclose(t.get("batteries")[0, 0], v[2])
    assert np.allclose(t.get("glass doors")[0, 0], v[3])


def test_out_of_vocabulary_is_none():
    t, _ = _table()
    assert t.get("giraffe") is None
    assert "giraffe" not in t and "chair" in t


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        VocabTable(["a", "b"], np.zeros((3, 512), np.float32))

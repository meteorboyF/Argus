import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_stereo", Path(__file__).parents[1] / "scripts" / "calibrate_stereo.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
normalize_corners = _MODULE.normalize_corners
COMMON_BOARDS = _MODULE.COMMON_BOARDS


def grid(cols, rows, step=20.0, origin=(100.0, 80.0)):
    ys, xs = np.mgrid[0:rows, 0:cols]
    pts = np.stack([origin[0] + xs * step, origin[1] + ys * step], axis=-1).astype(np.float32)
    return pts.reshape(-1, 1, 2)


def test_chess_board_is_a_known_layout():
    assert (7, 7) in COMMON_BOARDS


def test_canonical_order_is_unchanged():
    c = grid(7, 7)
    assert np.array_equal(normalize_corners(c, (7, 7)), c)


def test_rotated_180_detection_is_reordered():
    c = grid(7, 7)
    assert np.array_equal(normalize_corners(c[::-1].copy(), (7, 7)), c)


def test_transposed_square_detection_is_reordered():
    c = grid(7, 7)
    transposed = c.reshape(7, 7, 2).transpose(1, 0, 2).reshape(-1, 1, 2).copy()
    assert np.array_equal(normalize_corners(transposed, (7, 7)), c)


def test_rectangular_board_is_only_flipped_never_transposed():
    c = grid(9, 6)
    flipped = c.reshape(6, 9, 2)[:, ::-1].reshape(-1, 1, 2).copy()
    assert np.array_equal(normalize_corners(flipped, (9, 6)), c)

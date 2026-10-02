"""Machine-readable on-screen timecode for measuring true stereo exposure skew.

The two AR0234 cameras free-run without a shared trigger, so arrival time at
the host only approximates when each frame was exposed. To measure the real
offset, the monitor shows a millisecond timestamp encoded as a grid of black
and white cells framed by four ArUco markers. Each camera frame is decoded
back to the timestamp that was on screen when it was exposed.

Layout (canonical units, 1 unit = one cell):

    [M0]  b0  b1 ... b8   [M1]
    [M2]  b9 ...     b17  [M3]

Bits 0-15 hold the Gray code of (t_ms mod 65536). Gray code means a frame
exposed while the screen was mid-update is off by at most one LSB instead of
an arbitrary value. Bit 16 is even parity over bits 0-15 and bit 17 is its
complement, which rejects most torn or unreadable frames.

Resolution is bounded by the monitor refresh (16.7 ms at 60 Hz); the decoded
difference between left and right is averaged over many pairs, whose phase
relative to the refresh drifts, to estimate the mean offset below one refresh.
"""
from __future__ import annotations

import cv2
import numpy as np

BITS = 16
COLS = 9
ROWS = 2
MARKER_IDS = (0, 1, 2, 3)
_DICT = cv2.aruco.DICT_4X4_50

# Geometry in cell units. Markers are 2x2 cells, cells sit between them.
MARKER = 2.0
GAP = 0.5
GRID_X0 = MARKER + GAP
GRID_Y0 = 0.0
WIDTH = GRID_X0 + COLS + GAP + MARKER
HEIGHT = 2.0  # == ROWS == MARKER


def gray_encode(n: int) -> int:
    return n ^ (n >> 1)


def gray_decode(g: int) -> int:
    n = g
    shift = 1
    while (g >> shift) > 0:
        n ^= g >> shift
        shift += 1
    return n


def encode_bits(t_ms: int) -> list[int]:
    g = gray_encode(int(t_ms) % (1 << BITS))
    bits = [(g >> i) & 1 for i in range(BITS)]
    parity = sum(bits) & 1
    return bits + [parity, 1 - parity]


def decode_bits(bits) -> int | None:
    bits = [int(b) for b in bits]
    if len(bits) != BITS + 2:
        return None
    data, parity, check = bits[:BITS], bits[BITS], bits[BITS + 1]
    if parity == check or (sum(data) & 1) != parity:
        return None
    g = sum(b << i for i, b in enumerate(data))
    return gray_decode(g)


def marker_centres() -> dict[int, tuple[float, float]]:
    """Canonical centres of the four markers, in cell units."""
    right = WIDTH - MARKER / 2
    return {0: (MARKER / 2, 0.5), 1: (right, 0.5),
            2: (MARKER / 2, 1.5), 3: (right, 1.5)}


def _marker_image(marker_id: int, px: int) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(_DICT)
    return cv2.aruco.generateImageMarker(d, marker_id, px)


def render(t_ms: int, width: int = 1920, height: int = 1080,
           show_digits: bool = True) -> np.ndarray:
    """Full-screen pattern for timestamp `t_ms` (white background).

    Markers are drawn 1 cell tall and stacked so rows 0/1 each have a marker at
    both ends; each marker therefore spans exactly one row (centres above)."""
    cell = int(min((width - 80) / WIDTH, (height * 0.6) / HEIGHT))
    img = np.full((height, width, 3), 255, np.uint8)
    ox = (width - int(WIDTH * cell)) // 2
    oy = (height - int(HEIGHT * cell)) // 2
    mpx = cell  # one marker per row: 1x1 cell visual, centred on its row
    quiet = int(cell * 0.15)
    for mid, (cx, cy) in marker_centres().items():
        m = _marker_image(mid, mpx - 2 * quiet)
        x = int(ox + cx * cell - (mpx - 2 * quiet) / 2)
        y = int(oy + cy * cell - (mpx - 2 * quiet) / 2)
        img[y:y + m.shape[0], x:x + m.shape[1]] = m[..., None]
    for i, b in enumerate(encode_bits(t_ms)):
        r, c = divmod(i, COLS)
        x0 = int(ox + (GRID_X0 + c) * cell)
        y0 = int(oy + (GRID_Y0 + r) * cell)
        if b:
            continue  # white cell: background already white
        pad = max(1, cell // 20)
        cv2.rectangle(img, (x0 + pad, y0 + pad), (x0 + cell - pad, y0 + cell - pad),
                      (0, 0, 0), -1)
    if show_digits:
        cv2.putText(img, f"{int(t_ms) % 100000:05d} ms", (40, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5, cv2.LINE_AA)
    return img


_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        params = cv2.aruco.DetectorParameters()
        _DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(_DICT), params)
    return _DETECTOR


def decode_frame(frame: np.ndarray) -> int | None:
    """Timestamp (mod 65536 ms) shown in `frame`, or None if unreadable."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None:
        return None
    found = {}
    for c, i in zip(corners, ids.ravel()):
        if int(i) in MARKER_IDS:
            found[int(i)] = c.reshape(4, 2).mean(axis=0)
    if len(found) < 4:
        return None
    canon = marker_centres()
    src = np.float32([canon[i] for i in MARKER_IDS])
    dst = np.float32([found[i] for i in MARKER_IDS])
    H = cv2.getPerspectiveTransform(src, dst)
    pts = []
    for i in range(BITS + 2):
        r, c = divmod(i, COLS)
        pts.append((GRID_X0 + c + 0.5, GRID_Y0 + r + 0.5))
    img_pts = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)
    h, w = gray.shape
    vals = []
    for x, y in img_pts:
        xi, yi = int(round(x)), int(round(y))
        if not (2 <= xi < w - 2 and 2 <= yi < h - 2):
            return None
        vals.append(float(gray[yi - 2:yi + 3, xi - 2:xi + 3].mean()))
    vals = np.array(vals)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 30:
        return None
    bits = (vals > (lo + hi) / 2).astype(int)
    return decode_bits(bits)


def wrap_diff_ms(a: int, b: int) -> int:
    """a - b on the 16-bit ms ring, mapped to [-32768, 32767]."""
    d = (int(a) - int(b)) % (1 << BITS)
    return d - (1 << BITS) if d >= (1 << (BITS - 1)) else d

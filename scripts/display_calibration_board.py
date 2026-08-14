#!/usr/bin/env python3
"""Display a physically scaled checkerboard for headless stereo calibration.

The monitor's active pixel resolution and EDID-reported physical dimensions are
read from xrandr. Run this full-screen on the Jetson monitor, then run
calibrate_stereo.py --headless in another terminal while moving the rigid camera
rig through different positions and angles in front of the display.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess

import cv2
import numpy as np


XRANDR_RE = re.compile(
    r"connected(?: primary)? (\d+)x(\d+)\+\d+\+\d+.*? (\d+)mm x (\d+)mm")


def active_display() -> tuple[int, int, int, int]:
    env = os.environ.copy()
    output = subprocess.check_output(["xrandr", "--query"], text=True, env=env)
    for line in output.splitlines():
        match = XRANDR_RE.search(line)
        if match:
            return tuple(map(int, match.groups()))
    raise RuntimeError("Could not read active display resolution/size from xrandr")


def make_board(screen_w: int, screen_h: int, width_mm: int, height_mm: int,
               inner_cols: int, inner_rows: int, square_mm: float,
               reserve_left: int = 640) -> tuple[np.ndarray, int, int]:
    px_per_mm_x = screen_w / width_mm
    px_per_mm_y = screen_h / height_mm
    square_px_x = round(square_mm * px_per_mm_x)
    square_px_y = round(square_mm * px_per_mm_y)
    squares_x, squares_y = inner_cols + 1, inner_rows + 1
    board_w, board_h = squares_x * square_px_x, squares_y * square_px_y
    target_w = screen_w - reserve_left
    if board_w > target_w - 40 or board_h > screen_h - 100:
        raise ValueError(
            f"{square_mm:g} mm squares make a {board_w}x{board_h}px board, too large for "
            f"the {target_w}x{screen_h}px target area; reduce --square-mm or "
            "--reserve-left")
    image = np.full((screen_h, screen_w, 3), 127, dtype=np.uint8)
    # Keep the target entirely out of the live-preview region. The preview may
    # contain a recursive image of this monitor, but it can no longer occlude
    # checkerboard corners or become part of the target itself.
    image[:, :reserve_left] = 45
    x0 = reserve_left + (target_w - board_w) // 2
    y0 = (screen_h - board_h) // 2
    for row in range(squares_y):
        for col in range(squares_x):
            color = 255 if (row + col) % 2 == 0 else 0
            x1, y1 = x0 + col * square_px_x, y0 + row * square_px_y
            x2, y2 = x1 + square_px_x, y1 + square_px_y
            cv2.rectangle(image, (x1, y1), (x2, y2), (color, color, color), -1)
    cv2.putText(image, "LIVE PREVIEW AREA", (25, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (180, 180, 180), 2, cv2.LINE_AA)
    message = (f"ARGUS calibration target: {inner_cols}x{inner_rows} inner corners, "
               f"{square_mm:g} mm physical squares — Q/Esc to close")
    cv2.putText(image, message, (20, screen_h - 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return image, square_px_x, square_px_y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cols", type=int, default=9, help="inner columns")
    parser.add_argument("--rows", type=int, default=6, help="inner rows")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--reserve-left", type=int, default=640,
                        help="pixels reserved for the live stereo preview")
    args = parser.parse_args()

    screen_w, screen_h, width_mm, height_mm = active_display()
    board, px_x, px_y = make_board(screen_w, screen_h, width_mm, height_mm,
                                  args.cols, args.rows, args.square_mm,
                                  args.reserve_left)
    print(f"Display: {screen_w}x{screen_h}, EDID {width_mm}x{height_mm} mm")
    print(f"Rendered square: {px_x}x{px_y} px = {args.square_mm:g}x{args.square_mm:g} mm")
    print("Run in another terminal:")
    print(f"  python3 scripts/calibrate_stereo.py --cols {args.cols} "
          f"--rows {args.rows} --square-mm {args.square_mm:g} "
          "--preview-scale 0.45 --preview-x 10 --preview-y 90")
    window = "ARGUS on-screen calibration target"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        cv2.imshow(window, board)
        if cv2.waitKey(50) & 0xFF in (27, ord("q"), ord("Q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

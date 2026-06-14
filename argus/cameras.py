"""Camera capture for the three ARGUS cameras.

- 2x Arducam AR0234 global-shutter -> stereo pair (depth)
- 1x Arducam IMX477P wide -> scene camera (grounding + agent snapshots)

On the Jetson these are V4L2 UVC devices. We open them with OpenCV and, where
possible, request MJPG to keep USB bandwidth manageable when three cameras share
a controller. Stereo frames are captured grab()-then-retrieve() to minimise the
inter-camera time skew.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraConfig


def _open(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    # cv2.CAP_V4L2 is the right backend on the Jetson (Linux). On Windows this
    # falls back gracefully; for PC testing use argus_pc_test.ipynb instead.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)  # last-resort default backend
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab the freshest frame
    return cap


@dataclass
class StereoFrame:
    left: np.ndarray
    right: np.ndarray
    skew_ms: float
    ts: float


class CameraRig:
    """Owns all three cameras. Thread-safe wide-frame snapshot via a background
    reader so the slow loop always gets a recent frame without contention."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.left = _open(cfg.left_index, cfg.stereo_width, cfg.stereo_height, cfg.stereo_fps)
        self.right = _open(cfg.right_index, cfg.stereo_width, cfg.stereo_height, cfg.stereo_fps)
        self.wide = _open(cfg.wide_index, cfg.wide_width, cfg.wide_height, cfg.wide_fps)

        for name, cap in [("left", self.left), ("right", self.right), ("wide", self.wide)]:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {name} camera. Check indices with v4l2-ctl --list-devices")

        self._wide_lock = threading.Lock()
        self._wide_frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._wide_thread = threading.Thread(target=self._wide_loop, daemon=True)
        self._wide_thread.start()

    def _wide_loop(self):
        while not self._stop.is_set():
            ok, frame = self.wide.read()
            if ok:
                with self._wide_lock:
                    self._wide_frame = frame
            else:
                time.sleep(0.01)

    def get_wide_frame(self) -> np.ndarray | None:
        """Latest scene-camera frame (BGR), or None if not ready yet."""
        with self._wide_lock:
            return None if self._wide_frame is None else self._wide_frame.copy()

    def get_stereo_pair(self) -> StereoFrame | None:
        """Synchronized-ish left/right capture with measured skew."""
        if not (self.left.grab()):
            return None
        t_l = time.perf_counter()
        if not (self.right.grab()):
            return None
        t_r = time.perf_counter()
        okL, fL = self.left.retrieve()
        okR, fR = self.right.retrieve()
        if not (okL and okR):
            return None
        return StereoFrame(left=fL, right=fR, skew_ms=(t_r - t_l) * 1000.0, ts=t_l)

    def release(self):
        self._stop.set()
        if self._wide_thread.is_alive():
            self._wide_thread.join(timeout=1.0)
        for cap in (self.left, self.right, self.wide):
            cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

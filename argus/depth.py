"""Stereo depth estimation.

Two backends:
  - "sgbm":     OpenCV StereoSGBM, CPU. Always available, no model needed.
                Good enough for the geometric safety reflex.
  - "raft_trt": RAFT-Stereo TensorRT engine (built on-device). Higher quality,
                used when present. Falls back to SGBM if the engine is missing.

Depth (metres) = focal_px * baseline_m / disparity. focal_px and baseline_m come
from stereo calibration (config/stereo_calib.npz). Without calibration the
defaults give a rough, uncalibrated scale — fine for relative obstacle warnings,
not for precise distances.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .config import DepthConfig


class DepthEstimator:
    def __init__(self, cfg: DepthConfig):
        self.cfg = cfg
        self.backend = cfg.backend
        self._trt = None

        if self.backend == "raft_trt" and os.path.exists(cfg.raft_engine):
            try:
                from .trt_runner import TRTRunner
                self._trt = TRTRunner(cfg.raft_engine)
            except Exception as e:  # noqa: BLE001 — never let depth crash the rig
                print(f"[depth] RAFT TensorRT load failed ({e}); falling back to SGBM")
                self.backend = "sgbm"
        else:
            self.backend = "sgbm"

        if self.backend == "sgbm":
            self._sgbm = cv2.StereoSGBM_create(
                minDisparity=cfg.min_disparity,
                numDisparities=cfg.num_disparities,
                blockSize=cfg.block_size,
                P1=8 * 3 * cfg.block_size ** 2,
                P2=32 * 3 * cfg.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
            )

    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        if self.backend == "raft_trt" and self._trt is not None:
            return self._raft_disparity(left_bgr, right_bgr)
        gl = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        # SGBM returns fixed-point disparity scaled by 16.
        return self._sgbm.compute(gl, gr).astype(np.float32) / 16.0

    def _raft_disparity(self, left_bgr, right_bgr) -> np.ndarray:
        h, w = 480, 640
        def prep(img):
            img = cv2.resize(img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return np.transpose(img, (2, 0, 1))[None]  # NCHW
        out = self._trt.infer({"left": prep(left_bgr), "right": prep(right_bgr)})
        disp = list(out.values())[0].squeeze()
        return np.abs(disp).astype(np.float32)

    def depth_map(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Metric depth in metres (inf where disparity <= 0)."""
        disp = self.disparity(left_bgr, right_bgr)
        with np.errstate(divide="ignore"):
            depth = (self.cfg.focal_px * self.cfg.baseline_m) / disp
        depth[disp <= 0] = np.inf
        return depth

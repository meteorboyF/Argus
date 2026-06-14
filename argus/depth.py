"""Stereo depth estimation.

Two backends:
  - "sgbm":     OpenCV StereoSGBM, CPU. Always available, no model needed.
  - "raft_trt": RAFT-Stereo TensorRT engine (built on-device). Higher quality.
                Falls back to SGBM if the engine is missing.

CALIBRATION-AWARE. The ARGUS cameras are mounted on the curved sides of the
goggles, toed outward and non-coplanar — so raw disparity is meaningless until
the images are rectified. If a calibration file (config/stereo_calib.npz from
scripts/calibrate_stereo.py) is present we:
  1. remap (undistort + rectify) the left/right frames so epipolar lines align,
  2. compute disparity on the rectified pair,
  3. reproject to 3D with the Q matrix -> true metric depth that accounts for
     however the cameras were physically mounted.

Without a calibration file we fall back to the naive
depth = focal_px * baseline_m / disparity, which is only a rough relative scale
(fine for "something is close" warnings, not for accurate distances). The
runtime prints a clear warning in that case.
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
        self._calib = None
        self._load_calibration()

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

    # ------------------------------------------------------------------ calib
    def _load_calibration(self):
        path = self.cfg.calibration_file
        if not path or not os.path.exists(path):
            print(f"[depth] No calibration at {path!r} — using uncalibrated "
                  "baseline/focal scale. Run scripts/calibrate_stereo.py for "
                  "accurate, mounting-aware depth.")
            return
        try:
            data = np.load(path, allow_pickle=True)
            self._calib = {
                "map1x": data["map1x"], "map1y": data["map1y"],
                "map2x": data["map2x"], "map2y": data["map2y"],
                "Q": data["Q"],
                "size": tuple(int(v) for v in data["image_size"]),
            }
            # Prefer calibrated scalars if present (kept for the fallback path too).
            if "baseline_m" in data:
                self.cfg.baseline_m = float(data["baseline_m"])
            if "focal_px" in data:
                self.cfg.focal_px = float(data["focal_px"])
            toe = float(data["toe_angle_deg"]) if "toe_angle_deg" in data else float("nan")
            print(f"[depth] Loaded calibration {path} "
                  f"(baseline {self.cfg.baseline_m*100:.1f} cm, toe {toe:.1f} deg). "
                  "Rectification active.")
        except Exception as e:  # noqa: BLE001
            print(f"[depth] Failed to read calibration ({e}); uncalibrated fallback.")
            self._calib = None

    @property
    def calibrated(self) -> bool:
        return self._calib is not None

    def _rectify_pair(self, left_bgr, right_bgr):
        """Undistort + rectify so the cameras behave as a parallel pair."""
        c = self._calib
        # Calibration maps were built at the calibration resolution; resize if the
        # live frames differ so the maps still apply.
        if left_bgr.shape[1::-1] != c["size"]:
            left_bgr = cv2.resize(left_bgr, c["size"])
            right_bgr = cv2.resize(right_bgr, c["size"])
        lr = cv2.remap(left_bgr, c["map1x"], c["map1y"], cv2.INTER_LINEAR)
        rr = cv2.remap(right_bgr, c["map2x"], c["map2y"], cv2.INTER_LINEAR)
        return lr, rr

    # ------------------------------------------------------------------ disparity
    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        if self._calib is not None:
            left_bgr, right_bgr = self._rectify_pair(left_bgr, right_bgr)
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

    # ------------------------------------------------------------------ depth
    def depth_map(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Metric depth in metres (inf where invalid).

        With calibration: rectify -> disparity -> reprojectImageTo3D(Q) -> Z.
        This is correct for the actual mounting geometry (toe-out, baseline).
        Without calibration: rough depth = focal*baseline / disparity.
        """
        disp = self.disparity(left_bgr, right_bgr)

        if self._calib is not None:
            pts3d = cv2.reprojectImageTo3D(disp, self._calib["Q"])
            depth = pts3d[:, :, 2].astype(np.float32)
            # Invalid where disparity <= 0 or reprojection blew up.
            depth[disp <= 0] = np.inf
            depth[~np.isfinite(depth)] = np.inf
            depth[depth <= 0] = np.inf
            return depth

        with np.errstate(divide="ignore"):
            depth = (self.cfg.focal_px * self.cfg.baseline_m) / disp
        depth[disp <= 0] = np.inf
        return depth

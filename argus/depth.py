"""Stereo depth estimation.

Two backends:
  - "sgbm":     OpenCV StereoSGBM, CPU. Always available, no model needed.
                Matching runs on images downscaled by depth.fast_downscale so the
                10 Hz fast loop holds rate on the Orin Nano CPU; disparity is
                rescaled to full-resolution pixel units so Q stays valid.
  - "raft_trt": RAFT-Stereo TensorRT engine (built on-device). Higher quality.
                Production fails closed if the engine is missing or cannot load.
  - "cuda_sad": deterministic CUDA block matching, compiled for Orin SM 8.7.
                This is the default production backend.

CALIBRATION-AWARE. The ARGUS cameras are mounted on the curved sides of the
goggles, toed outward and non-coplanar — so raw disparity is meaningless until
the images are rectified. If a calibration file (config/stereo_calib.npz from
scripts/calibrate_stereo.py) is present we:
  1. remap (undistort + rectify) the left/right frames so epipolar lines align,
  2. compute disparity on the rectified pair,
  3. reproject to 3D with the Q matrix -> true metric depth that accounts for
     however the cameras were physically mounted.

Without a calibration file we expose a naive focal/baseline map for diagnostics
only. The orchestrator will not turn it into metric speech or danger decisions.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .calib_health import CalibrationMonitor
from .config import DepthConfig


class DepthEstimator:
    def __init__(self, cfg: DepthConfig):
        self.cfg = cfg
        self.backend = cfg.backend
        self._trt = None
        self._cuda_matcher = None
        self._calib = None
        self._load_calibration()
        self.health = CalibrationMonitor(cfg)

        if self.backend == "cuda_sad":
            try:
                from .cuda_stereo import CudaStereoMatcher
                self._cuda_matcher = CudaStereoMatcher()
            except Exception as e:  # noqa: BLE001
                if not cfg.allow_cpu_fallback:
                    self.health.stop()
                    raise RuntimeError(
                        "Production GPU depth failed to initialise; CPU fallback "
                        f"is disabled: {e}") from e
                print(f"[depth] diagnostic fallback after CUDA failure ({e})")
                self.backend = "sgbm"
        elif self.backend == "raft_trt":
            if not os.path.exists(cfg.raft_engine):
                if not cfg.allow_cpu_fallback:
                    self.health.stop()
                    raise RuntimeError(
                        f"Production GPU depth engine is missing: {cfg.raft_engine}. "
                        "CPU SGBM fallback is disabled.")
                print(f"[depth] diagnostic fallback: missing {cfg.raft_engine}; using CPU SGBM")
                self.backend = "sgbm"
            else:
                try:
                    from .trt_runner import TRTRunner
                    self._trt = TRTRunner(cfg.raft_engine)
                except Exception as e:  # noqa: BLE001
                    if not cfg.allow_cpu_fallback:
                        self.health.stop()
                        raise RuntimeError(
                            "Production GPU depth failed to initialise; CPU fallback "
                            f"is disabled: {e}") from e
                    print(f"[depth] diagnostic fallback after TensorRT failure ({e})")
                    self.backend = "sgbm"
        elif self.backend == "sgbm":
            if not cfg.allow_cpu_fallback:
                self.health.stop()
                raise RuntimeError(
                    "CPU SGBM is diagnostic-only. Set depth.backend=raft_trt for "
                    "production, or explicitly enable allow_cpu_fallback on the bench.")
        else:
            self.health.stop()
            raise ValueError(f"Unknown depth backend: {self.backend!r}")

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
            print(f"[depth] No calibration at {path!r} — diagnostic disparity "
                  "only; metric safety speech is disabled. Run "
                  "scripts/calibrate_stereo.py for mounting-aware depth.")
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
        """Disparity in FULL-resolution pixel units (so Q stays valid), even when
        SGBM matching runs on downscaled images for fast-loop speed."""
        if self._calib is not None:
            left_bgr, right_bgr = self._rectify_pair(left_bgr, right_bgr)
            # Watch for calibration drift on the rectified pair. Rate-limited
            # internally, so this is cheap to call every frame. Only meaningful
            # when calibrated — without a calibration there is no rectification
            # to have drifted.
            self.health.submit(left_bgr, right_bgr)
        if self.backend == "raft_trt" and self._trt is not None:
            return self._raft_disparity(left_bgr, right_bgr)
        if self.backend == "cuda_sad" and self._cuda_matcher is not None:
            return self._cuda_disparity(left_bgr, right_bgr)
        gl = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        s = max(1, int(self.cfg.fast_downscale))
        if s > 1:
            h, w = gl.shape
            gl = cv2.resize(gl, (w // s, h // s), interpolation=cv2.INTER_AREA)
            gr = cv2.resize(gr, (w // s, h // s), interpolation=cv2.INTER_AREA)
        # SGBM returns fixed-point disparity scaled by 16.
        disp = self._sgbm.compute(gl, gr).astype(np.float32) / 16.0
        if s > 1:
            # Invalid pixels (disp < 0) must not be blended into neighbours.
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_NEAREST) * s
        return disp

    def _cuda_disparity(self, left_bgr, right_bgr) -> np.ndarray:
        gl = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        scale = max(1, int(self.cfg.fast_downscale))
        full_h, full_w = gl.shape
        if scale > 1:
            size = (full_w // scale, full_h // scale)
            gl = cv2.resize(gl, size, interpolation=cv2.INTER_AREA)
            gr = cv2.resize(gr, size, interpolation=cv2.INTER_AREA)
        disparity = self._cuda_matcher.compute(
            gl, gr,
            max_disparity=max(2, int(self.cfg.num_disparities) // scale),
            radius=max(1, int(self.cfg.block_size) // 2),
            uniqueness_percent=int(self.cfg.sad_uniqueness_percent),
        )
        if scale > 1:
            disparity = cv2.resize(
                disparity, (full_w, full_h), interpolation=cv2.INTER_NEAREST) * scale
        return disparity.astype(np.float32, copy=False)

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

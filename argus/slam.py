"""Stereo visual odometry: pose, trail and tracking quality for the fast loop.

This is the first SLAM backend for ARGUS. It is deliberately small and
inspectable rather than a full mapping system:

  1. ORB features on the rectified left image, matched to the rectified right
     image along the same row (rectification makes epipolar lines horizontal),
     which triangulates a sparse 3D point cloud in the left camera frame.
  2. Those left-image keypoints are tracked into the next left frame with
     pyramidal Lucas-Kanade optical flow.
  3. PnP with RANSAC between the previous frame's 3D points and the current
     frame's 2D tracks gives the camera motion; poses are chained.

What it provides to the rest of the system:
  - pose (x, z, yaw) in the frame where ARGUS started, and the trail for the
    dashboard's top-down view,
  - tracking quality (inlier ratio, feature count) so consumers know when the
    estimate is untrustworthy,
  - ego speed, so the approach rule can tell "I am walking toward a wall"
    from "something is driving toward me".

Requires stereo calibration. Without it the backend reports "off"; there is no
uncalibrated fallback because scale and rectification would both be fiction.

The interface (`submit`, `state`) is what a heavier backend (ORB-SLAM3,
cuVSLAM) would implement later; nothing else in ARGUS depends on the internals.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import SlamConfig


@dataclass
class SlamState:
    tracking: bool = False
    quality: float = 0.0          # RANSAC inlier ratio of the last solve
    features: int = 0
    inliers: int = 0
    x_m: float = 0.0              # right (+) in the start frame
    z_m: float = 0.0              # forward (+) in the start frame
    yaw_deg: float = 0.0
    speed_mps: float = 0.0
    rate_hz: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)
    reason: str = "not started"


class StereoVisualOdometry:
    def __init__(self, cfg: SlamConfig, fx: float, fy: float, cx: float, cy: float,
                 baseline_m: float):
        self.cfg = cfg
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.baseline_m = baseline_m
        self._orb = cv2.ORB_create(nfeatures=cfg.max_features, fastThreshold=12)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._pose = np.eye(4)            # camera-to-world
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._prev_xyz: np.ndarray | None = None
        self._pending: tuple[np.ndarray, np.ndarray, float] | None = None
        self._lock = threading.Lock()
        self._state = SlamState()
        self._trail: collections.deque = collections.deque(maxlen=cfg.trail_points)
        self._positions: collections.deque = collections.deque(maxlen=12)
        self._proc_times: collections.deque = collections.deque(maxlen=20)
        self._stop = threading.Event()
        self._new = threading.Event()
        self._thread = threading.Thread(target=self._run, name="argus-slam", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ interface
    def submit(self, left_rect_bgr: np.ndarray, right_rect_bgr: np.ndarray, ts: float) -> None:
        """Hand the newest rectified pair to the worker. Never blocks."""
        with self._lock:
            self._pending = (left_rect_bgr, right_rect_bgr, ts)
        self._new.set()

    @property
    def state(self) -> SlamState:
        with self._lock:
            return SlamState(**{**self._state.__dict__, "trail": list(self._trail)})

    def stop(self):
        self._stop.set()
        self._new.set()
        self._thread.join(timeout=1.0)

    # ------------------------------------------------------------ worker
    def _run(self):
        while not self._stop.is_set():
            self._new.wait(0.5)
            self._new.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            started = time.perf_counter()
            try:
                self._step(*job)
            except Exception as exc:  # noqa: BLE001 — SLAM must never take the loop down
                with self._lock:
                    self._state.tracking = False
                    self._state.reason = f"error: {exc}"
            self._proc_times.append(time.perf_counter() - started)

    def _triangulate(self, gray_l: np.ndarray, gray_r: np.ndarray):
        """Sparse stereo: ORB on both images, row-constrained matching."""
        kp_l, des_l = self._orb.detectAndCompute(gray_l, None)
        kp_r, des_r = self._orb.detectAndCompute(gray_r, None)
        if des_l is None or des_r is None or len(kp_l) < 8 or len(kp_r) < 8:
            return None, None
        pts, xyz = [], []
        for m in self._matcher.match(des_l, des_r):
            (xl, yl), (xr, yr) = kp_l[m.queryIdx].pt, kp_r[m.trainIdx].pt
            d = xl - xr
            if abs(yl - yr) > self.cfg.max_row_error_px or d < self.cfg.min_disparity_px:
                continue
            z = self.fx * self.baseline_m / d
            if z > self.cfg.max_depth_m:
                continue
            pts.append((xl, yl))
            xyz.append(((xl - self.cx) * z / self.fx, (yl - self.cy) * z / self.fy, z))
        if len(pts) < 8:
            return None, None
        return (np.array(pts, dtype=np.float32).reshape(-1, 1, 2),
                np.array(xyz, dtype=np.float32))

    def _step(self, left_bgr: np.ndarray, right_bgr: np.ndarray, ts: float):
        gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        tracking, quality, inliers, n_feat = False, 0.0, 0, 0

        if self._prev_gray is not None and self._prev_pts is not None and len(self._prev_pts) >= 8:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray_l, self._prev_pts, None, winSize=(21, 21), maxLevel=3)
            ok = status.reshape(-1) == 1
            n_feat = int(ok.sum())
            if n_feat >= self.cfg.min_inliers:
                obj = self._prev_xyz[ok].astype(np.float64)
                img = nxt[ok].reshape(-1, 2).astype(np.float64)
                success, rvec, tvec, inl = cv2.solvePnPRansac(
                    obj, img, self.K, None, iterationsCount=100,
                    reprojectionError=self.cfg.reprojection_px, confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if success and inl is not None and len(inl) >= self.cfg.min_inliers:
                    inliers = int(len(inl))
                    quality = inliers / max(1, n_feat)
                    R, _ = cv2.Rodrigues(rvec)
                    # solvePnP gives prev-camera-frame -> current-camera-frame.
                    T_cur_prev = np.eye(4)
                    T_cur_prev[:3, :3], T_cur_prev[:3, 3] = R, tvec.reshape(3)
                    motion = np.linalg.inv(T_cur_prev)          # current pose in prev frame
                    step = float(np.linalg.norm(motion[:3, 3]))
                    if step <= self.cfg.max_step_m:
                        self._pose = self._pose @ motion
                        tracking = True

        pts, xyz = self._triangulate(gray_l, gray_r)
        self._prev_gray, self._prev_pts, self._prev_xyz = gray_l, pts, xyz

        x, z = float(self._pose[0, 3]), float(self._pose[2, 3])
        yaw = math.degrees(math.atan2(self._pose[0, 2], self._pose[2, 2]))
        self._positions.append((ts, x, z))
        speed = 0.0
        if len(self._positions) >= 4:
            t0, x0, z0 = self._positions[0]
            dt = ts - t0
            if dt > 0.05:
                speed = math.hypot(x - x0, z - z0) / dt
        if tracking and (not self._trail or math.hypot(x - self._trail[-1][0], z - self._trail[-1][1]) > 0.02):
            self._trail.append((x, z))
        rate = (len(self._proc_times) / sum(self._proc_times)) if self._proc_times else 0.0
        with self._lock:
            self._state = SlamState(
                tracking=tracking, quality=quality, features=n_feat, inliers=inliers,
                x_m=x, z_m=z, yaw_deg=yaw, speed_mps=speed, rate_hz=rate,
                reason="tracking" if tracking else
                       ("initialising" if pts is not None else "too few stereo features"))

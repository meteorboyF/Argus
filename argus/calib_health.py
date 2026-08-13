"""Runtime stereo calibration health check.

The problem this solves: calibration measures the exact rotation and translation
between the two AR0234s and bakes it into rectification maps. If a camera then
moves — a knocked ball joint, a flexed frame, a re-seated mount — the runtime has
no idea. It keeps applying the old rectification, SGBM still finds matches, and a
depth map still comes out. It is simply wrong, and wrong *systematically* rather
than noisily, so it reads as a confident measurement rather than a fault.

For a device that tells a blind user "Stop, obstacle very close", silent bad
depth is the failure mode that matters most.

How it is detected: after rectification, corresponding points in the left and
right images must lie on the same row — that is what rectification is for. So
sparse features are matched across the rectified pair and the median absolute
*vertical* offset is measured. On a good calibration that residual is well under
a pixel. As the geometry drifts it grows, and it needs no checkerboard, no user
action, and no knowledge of the scene.

Cost is kept off the safety path two ways: the check is sampled every
`health_interval_s` seconds rather than every frame, and the measurement itself
runs on its own worker thread, so the fast loop never pays for it.

This module only measures and reports. It deliberately does NOT disable the
safety loop on drift: degraded obstacle warnings still beat no obstacle
warnings. The orchestrator decides what to tell the user.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class CalibState(Enum):
    UNKNOWN = "unknown"      # not enough samples yet, or no calibration loaded
    OK = "ok"                # vertical residual within tolerance
    DEGRADED = "degraded"    # sustained residual above tolerance — re-calibrate


@dataclass
class CalibReading:
    vertical_px: float       # median |dy| across matched features
    matches: int
    when: float


class CalibrationMonitor:
    """Watches rectified stereo pairs for calibration drift.

    Feed it rectified pairs via `submit()`; it rate-limits internally, so calling
    it every frame is fine. Read `state` / `last` for the verdict.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = CalibState.UNKNOWN
        self.last: CalibReading | None = None
        self._orb = cv2.ORB_create(nfeatures=int(cfg.health_max_features))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._last_check = 0.0
        self._bad_streak = 0
        self._good_streak = 0

        # Feature matching costs tens of milliseconds — far too much to spend on
        # the fast loop's thread, which must never stall (that is the whole point
        # of the two-speed design). submit() therefore only hands off a pair; the
        # measurement happens here.
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._worker = threading.Thread(target=self._run, name="argus-calib", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ public
    def submit(self, left_rect: np.ndarray, right_rect: np.ndarray) -> bool:
        """Offer a rectified pair for checking. Non-blocking.

        Returns True if the pair was accepted for measurement. Safe to call on
        every frame: it rate-limits to health_interval_s and never does real work
        on the caller's thread.
        """
        if not self.cfg.health_check:
            return False
        now = time.monotonic()
        if now - self._last_check < self.cfg.health_interval_s:
            return False
        self._last_check = now
        with self._lock:
            if self._pending is not None:
                return False          # worker still busy with the previous pair
            # cv2.remap allocates fresh arrays per frame, so these are not
            # mutated behind our back and need no copy.
            self._pending = (left_rect, right_rect)
        self._wake.set()
        return True

    def stop(self):
        self._stopping = True
        self._wake.set()
        self._worker.join(timeout=2.0)

    def _run(self):
        while not self._stopping:
            if not self._wake.wait(0.5):
                continue
            self._wake.clear()
            with self._lock:
                pair = self._pending
            if pair is None or self._stopping:
                continue
            try:
                reading = self.measure(*pair)
                if reading is not None:
                    self.last = reading
                    self._update_state(reading)
            except Exception as e:  # noqa: BLE001 — diagnostics must never break depth
                print(f"[calib] health check failed: {e}")
            finally:
                with self._lock:
                    self._pending = None

    def measure(self, left_rect: np.ndarray, right_rect: np.ndarray) -> CalibReading | None:
        """Median absolute vertical offset of matched features, in pixels.

        Returns None when the scene has too little texture to judge — that is an
        absent measurement, not a passing one, and must not clear a drift state.
        """
        gl, gr, scale = _prep(left_rect, right_rect, self.cfg.health_downscale)
        kl, dl = self._orb.detectAndCompute(gl, None)
        kr, dr = self._orb.detectAndCompute(gr, None)
        if dl is None or dr is None or len(kl) < 2 or len(kr) < 2:
            return None

        matches = self._matcher.match(dl, dr)
        if len(matches) < self.cfg.health_min_matches:
            return None

        dy = np.array([abs(kl[m.queryIdx].pt[1] - kr[m.trainIdx].pt[1]) for m in matches],
                      dtype=np.float32)
        dx = np.array([kl[m.queryIdx].pt[0] - kr[m.trainIdx].pt[0] for m in matches],
                      dtype=np.float32)

        # Keep only plausible stereo correspondences. In a rectified pair the
        # left image sees a point further right than the right image does, so
        # dx > 0; negative or absurd dx is a mismatch, not a geometry error, and
        # including it would inflate the residual and cry wolf.
        keep = (dx > 0) & (dx < gl.shape[1] * 0.5)
        if int(keep.sum()) < self.cfg.health_min_matches:
            return None
        dy = dy[keep]

        # Median, not mean: a handful of surviving mismatches must not dominate.
        vertical_px = float(np.median(dy)) * scale
        return CalibReading(vertical_px=vertical_px, matches=int(keep.sum()), when=time.monotonic())

    # ---------------------------------------------------------------- internal
    def _update_state(self, reading: CalibReading):
        """Debounced with hysteresis, so one awkward scene cannot flip the verdict."""
        if reading.vertical_px > self.cfg.health_max_vertical_px:
            self._bad_streak += 1
            self._good_streak = 0
        else:
            self._good_streak += 1
            self._bad_streak = 0

        need = int(self.cfg.health_consecutive)
        if self._bad_streak >= need and self.state is not CalibState.DEGRADED:
            self.state = CalibState.DEGRADED
            print(f"[calib] DRIFT DETECTED — rectified rows are misaligned by "
                  f"{reading.vertical_px:.2f} px (limit {self.cfg.health_max_vertical_px:.2f}). "
                  "A camera has almost certainly moved. Depth distances are no longer "
                  "trustworthy; re-run scripts/calibrate_stereo.py.")
        elif self._good_streak >= need and self.state is not CalibState.OK:
            was = self.state
            self.state = CalibState.OK
            if was is CalibState.DEGRADED:
                print(f"[calib] alignment recovered ({reading.vertical_px:.2f} px). "
                      "If you did not re-calibrate, treat this as intermittent and "
                      "check the mounts.")


def _prep(left_bgr: np.ndarray, right_bgr: np.ndarray, downscale: int):
    """Grayscale + downscale both frames. Returns (left, right, scale_back)."""
    gl = left_bgr if left_bgr.ndim == 2 else cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gr = right_bgr if right_bgr.ndim == 2 else cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    s = max(1, int(downscale))
    if s > 1:
        h, w = gl.shape[:2]
        gl = cv2.resize(gl, (w // s, h // s), interpolation=cv2.INTER_AREA)
        gr = cv2.resize(gr, (w // s, h // s), interpolation=cv2.INTER_AREA)
    # Residuals are measured on the downscaled image, so scale back to
    # full-resolution pixels — otherwise the threshold means different things at
    # different downscale settings.
    return gl, gr, float(s)

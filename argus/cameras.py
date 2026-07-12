"""Camera capture for the three ARGUS cameras — with smart discovery.

- 2x Arducam AR0234 global-shutter -> stereo pair (depth)
- 1x Arducam IMX477P wide -> scene camera (grounding + agent snapshots)

Why discovery matters: V4L2 index assignment on the Jetson is NOT stable — the
same physical camera can be /dev/video0 on one boot and /dev/video2 on the next,
and each USB camera typically exposes two nodes (capture + metadata). Hardcoded
indices therefore break randomly. This module identifies cameras by:

  1. V4L2 device NAME (read from /sys/class/video4linux/videoN/name) matched
     against stereo_name_hint / wide_name_hint ("AR0234", "IMX477"),
  2. resolution grouping as a fallback (the two matching-resolution nodes are
     the stereo pair; the odd one out is the wide camera),
  3. USB port paths stored in the stereo calibration file, so LEFT vs RIGHT is
     resolved to the same physical camera the calibration was measured on, no
     matter how enumeration shuffles.

On the Jetson these are UVC devices opened via V4L2. MJPG is requested to keep
USB bandwidth manageable when three cameras share a controller. Stereo frames
are captured grab()-then-retrieve() to minimise the inter-camera time skew.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
@dataclass
class VideoNode:
    index: int
    name: str          # V4L2 device name ("" when unknown, e.g. on Windows)
    usb_port: str      # stable USB path like "1-2.3" ("" when unknown)
    width: int = 0     # probed default capture resolution
    height: int = 0
    works: bool = False  # opened AND delivered a frame


def _sysfs_nodes(max_index: int) -> list[VideoNode]:
    """Enumerate /dev/video* via sysfs (Linux/Jetson). Cheap — no device opens."""
    nodes = []
    base = Path("/sys/class/video4linux")
    if not base.exists():
        return [VideoNode(index=i, name="", usb_port="") for i in range(max_index)]
    for d in sorted(base.glob("video*")):
        try:
            idx = int(d.name[5:])
        except ValueError:
            continue
        if idx >= max_index:
            continue
        try:
            name = (d / "name").read_text().strip()
        except OSError:
            name = ""
        usb_port = ""
        try:
            # .../devices/platform/.../usb1/1-2/1-2.3/1-2.3:1.0/video4linux/videoN
            dev = (d / "device").resolve()
            usb_port = dev.name.split(":")[0]
        except OSError:
            pass
        nodes.append(VideoNode(index=idx, name=name, usb_port=usb_port))
    return nodes


def probe_cameras(max_index: int = 10, verbose: bool = True) -> list[VideoNode]:
    """Enumerate candidate nodes and verify each actually delivers frames.

    USB cameras expose extra metadata nodes that open but never return a frame;
    those are filtered out here.
    """
    nodes = _sysfs_nodes(max_index)
    working: list[VideoNode] = []
    for node in nodes:
        cap = cv2.VideoCapture(node.index, cv2.CAP_V4L2) if sys.platform.startswith("linux") \
            else cv2.VideoCapture(node.index)
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            node.works = True
            node.height, node.width = frame.shape[:2]
            working.append(node)
        cap.release()
    if verbose:
        for n in working:
            print(f"[cameras] /dev/video{n.index}: '{n.name}' "
                  f"{n.width}x{n.height} usb={n.usb_port or '?'}")
    return working


def _calib_ports(calibration_file: str) -> tuple[str, str] | None:
    """(left_usb_port, right_usb_port) recorded by calibrate_stereo.py, if any."""
    try:
        data = np.load(calibration_file, allow_pickle=True)
        lp, rp = str(data["left_port"]), str(data["right_port"])
        if lp and rp:
            return lp, rp
    except Exception:  # noqa: BLE001 — missing file/keys means no port info
        pass
    return None


def discover_rig(cfg: CameraConfig) -> tuple[int, int, int]:
    """Resolve (left_index, right_index, wide_index) from what is plugged in.

    Order of evidence:
      1. V4L2 device names vs stereo/wide name hints.
      2. Resolution grouping (two nodes sharing a resolution = stereo pair).
      3. USB ports from the calibration file decide left vs right.
    Falls back to the configured indices when discovery cannot decide.
    """
    nodes = probe_cameras(cfg.max_probe_index)
    if len(nodes) < 3:
        print(f"[cameras] discovery found only {len(nodes)} working camera(s); "
              f"falling back to configured indices "
              f"({cfg.left_index}/{cfg.right_index}/{cfg.wide_index}).")
        return cfg.left_index, cfg.right_index, cfg.wide_index

    stereo_hint = cfg.stereo_name_hint.lower()
    wide_hint = cfg.wide_name_hint.lower()
    stereo = [n for n in nodes if stereo_hint and stereo_hint in n.name.lower()]
    wide = [n for n in nodes if wide_hint and wide_hint in n.name.lower()]

    if len(stereo) != 2 or len(wide) < 1:
        # Fall back to resolution grouping.
        by_res: dict[tuple[int, int], list[VideoNode]] = {}
        for n in nodes:
            by_res.setdefault((n.width, n.height), []).append(n)
        pairs = [g for g in by_res.values() if len(g) == 2]
        if len(pairs) == 1:
            stereo = pairs[0]
            wide = [n for n in nodes if n not in stereo]
        else:
            print("[cameras] discovery ambiguous (names and resolutions inconclusive); "
                  "using configured indices. Set stereo_name_hint/wide_name_hint or "
                  "left/right/wide_index in argus.yaml.")
            return cfg.left_index, cfg.right_index, cfg.wide_index

    # Left vs right: match USB ports recorded at calibration time.
    ports = _calib_ports(cfg.calibration_file)
    left = right = None
    if ports:
        lp, rp = ports
        for n in stereo:
            if n.usb_port == lp:
                left = n
            elif n.usb_port == rp:
                right = n
    if left is None or right is None:
        if ports:
            print("[cameras] WARNING: calibration USB ports don't match the connected "
                  "cameras (cables moved?). Assuming index order for left/right — "
                  "re-run scripts/calibrate_stereo.py to re-bind and re-verify depth.")
        stereo_sorted = sorted(stereo, key=lambda n: n.index)
        left, right = stereo_sorted[0], stereo_sorted[1]

    wide_node = sorted(wide, key=lambda n: n.index)[0]
    print(f"[cameras] discovered: left=/dev/video{left.index} "
          f"right=/dev/video{right.index} wide=/dev/video{wide_node.index}")
    return left.index, right.index, wide_node.index


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
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
    reader so the slow loop always gets a recent frame without contention.
    Stereo cameras that stop delivering frames are reopened automatically."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        if cfg.auto_detect:
            self.left_index, self.right_index, self.wide_index = discover_rig(cfg)
        else:
            self.left_index, self.right_index, self.wide_index = (
                cfg.left_index, cfg.right_index, cfg.wide_index)

        self.left = _open(self.left_index, cfg.stereo_width, cfg.stereo_height, cfg.stereo_fps)
        self.right = _open(self.right_index, cfg.stereo_width, cfg.stereo_height, cfg.stereo_fps)
        self.wide = _open(self.wide_index, cfg.wide_width, cfg.wide_height, cfg.wide_fps)

        for name, cap in [("left", self.left), ("right", self.right), ("wide", self.wide)]:
            if not cap.isOpened():
                raise RuntimeError(
                    f"Failed to open {name} camera. Check connections and "
                    "v4l2-ctl --list-devices; override indices in argus.yaml if needed.")

        self._stereo_fail_since: float | None = None
        self._wide_lock = threading.Lock()
        self._wide_frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._wide_thread = threading.Thread(target=self._wide_loop, daemon=True)
        self._wide_thread.start()

    # ------------------------------------------------------------------ wide
    def _wide_loop(self):
        fail_since = None
        while not self._stop.is_set():
            ok, frame = self.wide.read()
            if ok:
                fail_since = None
                with self._wide_lock:
                    self._wide_frame = frame
            else:
                if fail_since is None:
                    fail_since = time.monotonic()
                elif (self.cfg.reconnect
                      and time.monotonic() - fail_since > self.cfg.reconnect_interval_s):
                    print("[cameras] wide camera stalled — reopening")
                    self.wide.release()
                    self.wide = _open(self.wide_index, self.cfg.wide_width,
                                      self.cfg.wide_height, self.cfg.wide_fps)
                    fail_since = None
                time.sleep(0.05)

    def get_wide_frame(self) -> np.ndarray | None:
        """Latest scene-camera frame (BGR), or None if not ready yet."""
        with self._wide_lock:
            return None if self._wide_frame is None else self._wide_frame.copy()

    # ------------------------------------------------------------------ stereo
    def get_stereo_pair(self) -> StereoFrame | None:
        """Synchronized-ish left/right capture with measured skew."""
        if not self.left.grab():
            self._stereo_failed()
            return None
        t_l = time.perf_counter()
        if not self.right.grab():
            self._stereo_failed()
            return None
        t_r = time.perf_counter()
        okL, fL = self.left.retrieve()
        okR, fR = self.right.retrieve()
        if not (okL and okR):
            self._stereo_failed()
            return None
        self._stereo_fail_since = None
        return StereoFrame(left=fL, right=fR, skew_ms=(t_r - t_l) * 1000.0, ts=t_l)

    def _stereo_failed(self):
        """Track sustained stereo failure and reopen both cameras if configured."""
        now = time.monotonic()
        if self._stereo_fail_since is None:
            self._stereo_fail_since = now
            return
        if self.cfg.reconnect and now - self._stereo_fail_since > self.cfg.reconnect_interval_s:
            print("[cameras] stereo pair stalled — reopening both")
            self.left.release()
            self.right.release()
            self.left = _open(self.left_index, self.cfg.stereo_width,
                              self.cfg.stereo_height, self.cfg.stereo_fps)
            self.right = _open(self.right_index, self.cfg.stereo_width,
                               self.cfg.stereo_height, self.cfg.stereo_fps)
            self._stereo_fail_since = None

    # ------------------------------------------------------------------ lifecycle
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

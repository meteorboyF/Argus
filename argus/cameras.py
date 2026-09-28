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
    link_mbps: int = 0   # negotiated USB link speed (5000 = USB 3, 480 = USB 2)


def usb_link_mbps(usb_port: str) -> int:
    """Negotiated link speed for a USB port path, 0 when unknown."""
    try:
        return int(float(Path(f"/sys/bus/usb/devices/{usb_port}/speed").read_text().strip()))
    except (OSError, ValueError):
        return 0


def hub_port_of(usb_port: str) -> str:
    """The physical hub port number, the part after the last dot.

    A USB 3 hub appears twice to Linux: once on the SuperSpeed bus and once on
    its High-Speed companion bus. The same physical port therefore shows up as
    "2-1.2" when the camera links at 5 Gbit/s and "1-2.2" when it only reaches
    USB 2. Binding on the hub port keeps the role stable across both.
    """
    return usb_port.rsplit(".", 1)[-1] if usb_port else ""


def find_port(nodes, wanted: str):
    """Exact USB path first, then the same hub port on the companion bus."""
    for node in nodes:
        if node.usb_port == wanted:
            return node
    hub = hub_port_of(wanted)
    same_hub = [n for n in nodes if hub and hub_port_of(n.usb_port) == hub]
    if len(same_hub) == 1:
        print(f"[cameras] {wanted} not present; using {same_hub[0].usb_port} "
              f"(same hub port {hub}, link {same_hub[0].link_mbps} Mbit/s)")
        return same_hub[0]
    return None


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
        nodes.append(VideoNode(index=idx, name=name, usb_port=usb_port,
                               link_mbps=usb_link_mbps(usb_port)))
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
                  f"{n.width}x{n.height} usb={n.usb_port or '?'} "
                  f"link={n.link_mbps or '?'} Mbit/s")
            if 0 < n.link_mbps < 5000:
                print(f"[cameras] WARNING: /dev/video{n.index} negotiated USB 2 "
                      "only; reseat the cable at both ends for full frame rate")
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


def order_stereo_nodes(stereo: list[VideoNode], cfg: CameraConfig,
                       calibration_ports: tuple[str, str] | None = None
                       ) -> tuple[VideoNode, VideoNode]:
    """Order two stereo nodes by stable physical identity.

    A completed calibration is authoritative. Before the first calibration,
    configured USB-port hints bind the mount-specific rotation and left/right
    role. `/dev/videoN` order is only the final fallback.
    """
    ports = calibration_ports
    if not ports and cfg.left_usb_port and cfg.right_usb_port:
        ports = (cfg.left_usb_port, cfg.right_usb_port)
    if ports:
        left = find_port(stereo, ports[0])
        right = find_port(stereo, ports[1])
        if left is not None and right is not None and left is not right:
            return left, right
        print("[cameras] WARNING: configured/calibrated stereo USB ports do not "
              "match the connected B0495 cameras; falling back to index order. "
              "Check cables and re-run calibration.")
    ordered = sorted(stereo, key=lambda node: node.index)
    return ordered[0], ordered[1]


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

    # Left vs right: calibration ports are authoritative; mount port hints make
    # the very first calibration stable before that file exists.
    ports = _calib_ports(cfg.calibration_file)
    left, right = order_stereo_nodes(stereo, cfg, ports)

    wide_node = find_port(wide, cfg.wide_usb_port) if cfg.wide_usb_port else None
    if wide_node is None:
        wide_node = sorted(wide, key=lambda n: n.index)[0]
    print(f"[cameras] discovered: left=/dev/video{left.index} "
          f"right=/dev/video{right.index} wide=/dev/video{wide_node.index}")
    return left.index, right.index, wide_node.index


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def transform_frame(frame: np.ndarray, rotation: int = 0,
                    flip_horizontal: bool = False,
                    flip_vertical: bool = False) -> np.ndarray:
    """Normalize a physically mounted camera into an upright BGR frame."""
    rotation = int(rotation) % 360
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotation not in rotations:
        raise ValueError(f"camera rotation must be 0/90/180/270, got {rotation}")
    if rotations[rotation] is not None:
        frame = cv2.rotate(frame, rotations[rotation])
    if flip_horizontal and flip_vertical:
        frame = cv2.flip(frame, -1)
    elif flip_horizontal:
        frame = cv2.flip(frame, 1)
    elif flip_vertical:
        frame = cv2.flip(frame, 0)
    return frame


def transformed_size(width: int, height: int, rotation: int) -> tuple[int, int]:
    """Return output (width, height) after the configured rotation."""
    return (height, width) if int(rotation) % 180 else (width, height)


def _open(index: int, width: int, height: int, fps: int,
          pixel_format: str = "YUYV") -> cv2.VideoCapture:
    # cv2.CAP_V4L2 is the right backend on the Jetson (Linux). On Windows this
    # falls back gracefully; archived PC experiments are not current guidance.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)  # last-resort default backend
    fourcc = (pixel_format or "YUYV").upper()
    if len(fourcc) != 4:
        raise ValueError(f"camera pixel_format must be four characters, got {fourcc!r}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
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


class _Reader:
    """Free-running capture thread for one camera with arrival timestamps.

    Serial grab()/grab() on two UVC cameras blocks the fast loop for up to a
    full frame period per camera and hides the true inter-camera skew. Each
    camera is read by its own thread; consumers take the newest frame and an
    honest arrival time, so the skew gate measures reality.
    """

    def __init__(self, name: str, index: int, width: int, height: int, fps: int,
                 pixel_format: str, rotation: int, flip_h: bool, flip_v: bool,
                 reconnect: bool, reconnect_interval_s: float):
        self.name, self.index = name, index
        self._open_args = (index, width, height, fps, pixel_format)
        self._xform = (rotation, flip_h, flip_v)
        self._reconnect = reconnect
        self._reconnect_interval_s = reconnect_interval_s
        self.cap = _open(*self._open_args)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open {name} camera (/dev/video{index}). Check connections "
                "and v4l2-ctl --list-devices; override indices in argus.yaml if needed.")
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._seq = 0
        self._fps = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)
        self._thread.start()

    def _run(self):
        fail_since = None
        rate_t, rate_n = time.perf_counter(), 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if ok and frame is not None:
                fail_since = None
                frame = transform_frame(frame, *self._xform)
                with self._lock:
                    self._frame, self._ts, self._seq = frame, now, self._seq + 1
                rate_n += 1
                if now - rate_t >= 1.0:
                    self._fps = rate_n / (now - rate_t)
                    rate_t, rate_n = now, 0
                continue
            if fail_since is None:
                fail_since = time.monotonic()
            elif self._reconnect and time.monotonic() - fail_since > self._reconnect_interval_s:
                print(f"[cameras] {self.name} camera stalled — reopening")
                self.cap.release()
                self.cap = _open(*self._open_args)
                fail_since = None
            time.sleep(0.02)

    @property
    def fps(self) -> float:
        """Delivered rate; reports 0 once no frame has arrived for 2 s so a
        camera whose read() is blocked (USB 2 starvation) shows as stalled
        instead of keeping its last healthy number."""
        with self._lock:
            ts = self._ts
        return 0.0 if time.perf_counter() - ts > 2.0 else self._fps

    @property
    def link_mbps(self) -> int:
        """Current negotiated USB link speed of this camera (0 when unknown).

        Read live from sysfs so a reseat shows on the dashboard without a
        restart: 5000 means USB 3, 480 means the plug only made USB 2 contact."""
        try:
            dev = Path(f"/sys/class/video4linux/video{self.index}/device").resolve()
            return int(float((dev.parent / "speed").read_text().strip()))
        except (OSError, ValueError):
            return 0

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            return self._frame, self._ts, self._seq

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


class CameraRig:
    """Owns all three cameras. Every camera has its own reader thread, so the
    fast loop and the slow loop always take the newest frame without blocking
    on USB, and stereo skew is measured from real arrival times."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        if cfg.auto_detect:
            self.left_index, self.right_index, self.wide_index = discover_rig(cfg)
        else:
            self.left_index, self.right_index, self.wide_index = (
                cfg.left_index, cfg.right_index, cfg.wide_index)

        common = (cfg.reconnect, cfg.reconnect_interval_s)
        self.left = _Reader("left", self.left_index, cfg.stereo_width, cfg.stereo_height,
                            cfg.stereo_fps, cfg.pixel_format, cfg.left_rotation,
                            cfg.left_flip_horizontal, cfg.left_flip_vertical, *common)
        self.right = _Reader("right", self.right_index, cfg.stereo_width, cfg.stereo_height,
                             cfg.stereo_fps, cfg.pixel_format, cfg.right_rotation,
                             cfg.right_flip_horizontal, cfg.right_flip_vertical, *common)
        self.wide = _Reader("wide", self.wide_index, cfg.wide_width, cfg.wide_height,
                            cfg.wide_fps, cfg.pixel_format, cfg.wide_rotation,
                            cfg.wide_flip_horizontal, cfg.wide_flip_vertical, *common)
        self._last_pair = (-1, -1)

    # ------------------------------------------------------------------ wide
    def get_wide_frame(self) -> np.ndarray | None:
        """Latest scene-camera frame (BGR), or None if not ready yet."""
        frame, _, _ = self.wide.latest()
        return None if frame is None else frame.copy()

    # ------------------------------------------------------------------ stereo
    def get_stereo_pair(self) -> StereoFrame | None:
        """Newest left/right frames with their real arrival skew.

        Returns None until both cameras have delivered, and None again when
        neither camera has produced a new frame since the last call, so a
        stalled pair is never re-evaluated as fresh."""
        fL, tL, sL = self.left.latest()
        fR, tR, sR = self.right.latest()
        if fL is None or fR is None or (sL, sR) == self._last_pair:
            return None
        self._last_pair = (sL, sR)
        return StereoFrame(left=fL, right=fR, skew_ms=abs(tR - tL) * 1000.0, ts=min(tL, tR))

    def rates(self) -> dict[str, float]:
        return {"left": self.left.fps, "right": self.right.fps, "wide": self.wide.fps}

    # ------------------------------------------------------------------ lifecycle
    def release(self):
        for reader in (self.left, self.right, self.wide):
            reader.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

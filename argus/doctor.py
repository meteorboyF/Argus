"""`python3 -m argus doctor`: rig bring-up checks with a spoken summary.

Answers, with measurements rather than assumptions:
  - are all three cameras present, and is each on SuperSpeed (5 Gbit/s)?
  - what frame rate does each camera actually deliver when all stream at once?
  - how far apart do left/right frames arrive (median, p95, % over the gate)?
  - free RAM and swap, llama-server health, Piper playback.

It saves a snapshot per camera to /tmp/argus_snap/ so orientation, focus and
framing can be checked by looking at the pictures. Sub-modes:

  --snap        timed capture of several snapshots while the rig is worn
  --skew-test   true exposure offset from an on-screen timecode (argus.timecode)
  --bandwidth   stereo fps matrix with the wide camera closed vs streaming
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cameras import _open, _sysfs_nodes, order_stereo_nodes, transform_frame, VideoNode
from .config import ArgusConfig

SNAP_DIR = Path("/tmp/argus_snap")
REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def capture_nodes(max_index: int = 16) -> list[VideoNode]:
    """Video capture nodes only (each UVC camera also exposes a metadata node,
    which sysfs marks with index 1)."""
    out = []
    for n in _sysfs_nodes(max_index):
        try:
            idx_attr = int(Path(f"/sys/class/video4linux/video{n.index}/index").read_text())
        except (OSError, ValueError):
            idx_attr = 0
        if idx_attr == 0:
            out.append(n)
    return out


def assign_roles(nodes: list[VideoNode], cfg: ArgusConfig) -> dict[str, VideoNode]:
    """Map present nodes to left/right/wide. Missing roles are simply absent."""
    cam = cfg.camera
    stereo = [n for n in nodes if cam.stereo_name_hint.lower() in n.name.lower()]
    wide = [n for n in nodes if cam.wide_name_hint.lower() in n.name.lower()]
    roles: dict[str, VideoNode] = {}
    if len(stereo) >= 2:
        left, right = order_stereo_nodes(stereo[:2], cam)
        roles["left"], roles["right"] = left, right
    elif len(stereo) == 1:
        # Name the lone camera by the port it sits on so the human knows which
        # one is missing.
        only = stereo[0]
        hub = only.usb_port.rsplit(".", 1)[-1]
        if hub == cam.right_usb_port.rsplit(".", 1)[-1]:
            roles["right"] = only
        else:
            roles["left"] = only
    if wide:
        roles["wide"] = wide[0]
    return roles


# ---------------------------------------------------------------------------
# Streaming probe
# ---------------------------------------------------------------------------
class StreamProbe:
    """Opens one camera, records every frame's arrival time and keeps the
    newest frame. Measures open latency (open call -> first frame)."""

    def __init__(self, role: str, node: VideoNode, width: int, height: int, fps: int,
                 pixel_format: str, rotation: int = 0, keep_frames: int = 0):
        self.role, self.node = role, node
        self.rotation = rotation
        self.arrivals: list[float] = []
        self.frames: list[tuple[float, np.ndarray]] = []
        self.keep_frames = keep_frames
        self.latest: np.ndarray | None = None
        self.latest_ts = 0.0
        self.seq = 0
        self.error = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        t0 = time.perf_counter()
        self.cap = _open(node.index, width, height, fps, pixel_format)
        self.opened = self.cap.isOpened()
        self.open_s = time.perf_counter() - t0
        self.first_frame_s: float | None = None
        self._t_open = t0
        self.actual = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       float(self.cap.get(cv2.CAP_PROP_FPS))) if self.opened else (0, 0, 0.0)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"probe-{role}")
        if self.opened:
            self._thread.start()
        else:
            self.error = "open failed"

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if self.first_frame_s is None:
                self.first_frame_s = now - self._t_open
            with self._lock:
                self.arrivals.append(now)
                self.latest, self.latest_ts, self.seq = frame, now, self.seq + 1
                if self.keep_frames:
                    self.frames.append((now, frame))
                    if len(self.frames) > self.keep_frames:
                        self.frames.pop(0)

    def newest(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            return self.latest, self.latest_ts, self.seq

    def reset_stats(self):
        with self._lock:
            self.arrivals.clear()
            self.frames.clear()

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            f = self.latest
        return None if f is None else transform_frame(f, self.rotation)

    def delivered_fps(self) -> float:
        with self._lock:
            a = list(self.arrivals)
        if len(a) < 2:
            return 0.0
        return (len(a) - 1) / (a[-1] - a[0])

    def max_gap_ms(self) -> float:
        with self._lock:
            a = np.array(self.arrivals)
        return float(np.diff(a).max() * 1000) if len(a) > 1 else float("inf")

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.cap.release()


def pair_skews_ms(left: list[float], right: list[float]) -> np.ndarray:
    """For every left arrival, |t_left - nearest t_right| in ms."""
    if not left or not right:
        return np.array([])
    r = np.asarray(right)
    out = []
    for t in left:
        j = np.searchsorted(r, t)
        cands = [r[k] for k in (j - 1, j) if 0 <= k < len(r)]
        out.append(min(abs(t - c) for c in cands) * 1000.0)
    return np.asarray(out)


def skew_stats(skews: np.ndarray, gate_ms: float) -> dict:
    if skews.size == 0:
        return {"n": 0}
    return {"n": int(skews.size),
            "median_ms": round(float(np.median(skews)), 2),
            "p95_ms": round(float(np.percentile(skews, 95)), 2),
            "max_ms": round(float(skews.max()), 2),
            "over_gate_pct": round(float((skews > gate_ms).mean() * 100), 1),
            "gate_ms": gate_ms}


# ---------------------------------------------------------------------------
# System checks
# ---------------------------------------------------------------------------
def meminfo() -> dict:
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        vals[k] = int(v.split()[0]) // 1024
    return {"total_mb": vals["MemTotal"], "available_mb": vals["MemAvailable"],
            "swap_total_mb": vals["SwapTotal"],
            "swap_used_mb": vals["SwapTotal"] - vals["SwapFree"]}


def oc_events() -> dict[str, int]:
    """Cumulative over-current throttle events per soctherm channel. The board
    drops clocks when total input current crosses ~5 A at 5 V; measurements
    taken while this counter rises are throttled and must say so."""
    out = {}
    for f in sorted(Path("/sys/class/hwmon").glob("hwmon*/oc*_event_cnt")):
        try:
            out[f.name.split("_")[0]] = int(f.read_text())
        except (OSError, ValueError):
            pass
    return out


def llama_health(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            body = r.read().decode(errors="replace")[:80]
            return r.status == 200, body
    except Exception as e:  # noqa: BLE001
        return False, f"not reachable ({type(e).__name__})"


@dataclass
class Row:
    name: str
    ok: bool | None          # None = informational / skipped
    detail: str
    data: dict = field(default_factory=dict)


def _rotation(cfg: ArgusConfig, role: str) -> int:
    return int(getattr(cfg.camera, f"{role}_rotation"))


def _open_probes(cfg: ArgusConfig, roles: dict[str, VideoNode], stereo_fps: int | None = None,
                 wide: bool = True, keep_frames: int = 0) -> dict[str, StreamProbe]:
    cam = cfg.camera
    probes = {}
    for role, node in roles.items():
        if role == "wide":
            if not wide:
                continue
            probes[role] = StreamProbe(role, node, cam.wide_width, cam.wide_height, cam.wide_fps,
                                       cam.pixel_format, _rotation(cfg, role), keep_frames)
        else:
            probes[role] = StreamProbe(role, node, cam.stereo_width, cam.stereo_height,
                                       stereo_fps or cam.stereo_fps, cam.pixel_format,
                                       _rotation(cfg, role), keep_frames)
    return probes


def save_snaps(probes: dict[str, StreamProbe], tag: str = "") -> list[str]:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for role, p in probes.items():
        f = p.snapshot()
        if f is None:
            continue
        path = SNAP_DIR / f"{role}{tag}.jpg"
        cv2.imwrite(str(path), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(str(path))
    return paths


def _wait_frames(probes: dict[str, StreamProbe], timeout: float = 5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and any(p.opened and p.latest is None for p in probes.values()):
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Main doctor
# ---------------------------------------------------------------------------
def run_doctor(cfg: ArgusConfig, seconds: float = 10.0, speak: bool = True,
               check_audio: bool = True) -> dict:
    from .guide import Guide
    guide = Guide(cfg.speech, "ARGUS doctor", speak=speak, show=False)
    rows: list[Row] = []
    try:
        nodes = capture_nodes()
        roles = assign_roles(nodes, cfg)
        expected = {"left": "B0495", "right": "B0495", "wide": "B0459"}
        for role, model in expected.items():
            n = roles.get(role)
            if n is None:
                rows.append(Row(f"{role} camera present", False,
                                f"{model} not found (port hint "
                                f"{getattr(cfg.camera, role + '_usb_port') or '?'})"))
            else:
                rows.append(Row(f"{role} camera present", True,
                                f"/dev/video{n.index} {n.name} @ {n.usb_port}"))
                rows.append(Row(f"{role} on SuperSpeed", n.link_mbps >= 5000,
                                f"{n.link_mbps} Mbit/s" + ("" if n.link_mbps >= 5000
                                                          else " (reseat the USB-C plug)")))
        probes = _open_probes(cfg, roles)
        try:
            _wait_frames(probes)
            for p in probes.values():
                p.reset_stats()
            time.sleep(seconds)
            for role, p in probes.items():
                target = cfg.camera.wide_fps if role == "wide" else cfg.camera.stereo_fps
                fps = p.delivered_fps()
                rows.append(Row(f"{role} fps ({seconds:.0f}s)", fps >= 0.9 * target,
                                f"{fps:.1f} of {target} fps, worst gap {p.max_gap_ms():.0f} ms, "
                                f"first frame {p.first_frame_s or float('nan'):.2f} s, "
                                f"mode {p.actual[0]}x{p.actual[1]}",
                                {"fps": round(fps, 2), "target": target,
                                 "max_gap_ms": round(p.max_gap_ms(), 1),
                                 "open_to_first_frame_s": p.first_frame_s}))
            if "left" in probes and "right" in probes:
                st = skew_stats(pair_skews_ms(probes["left"].arrivals, probes["right"].arrivals),
                                cfg.camera.max_skew_ms)
                ok = st.get("n", 0) > 0 and st["p95_ms"] <= cfg.camera.max_skew_ms
                rows.append(Row("stereo arrival skew", ok,
                                f"median {st.get('median_ms')} ms, p95 {st.get('p95_ms')} ms, "
                                f"{st.get('over_gate_pct')}% over {cfg.camera.max_skew_ms} ms gate",
                                st))
            else:
                rows.append(Row("stereo arrival skew", False, "needs both stereo cameras"))
            snaps = save_snaps(probes)
            rows.append(Row("snapshots", None, ", ".join(snaps) or "none"))
        finally:
            for p in probes.values():
                p.close()

        mem = meminfo()
        rows.append(Row("free RAM", mem["available_mb"] >= 1500,
                        f"{mem['available_mb']} MB available of {mem['total_mb']} MB, "
                        f"swap {mem['swap_used_mb']}/{mem['swap_total_mb']} MB", mem))
        oc = oc_events()
        rows.append(Row("over-current throttle events", None,
                        ", ".join(f"{k}={v}" for k, v in oc.items()) or "unavailable", oc))
        ok, body = llama_health(cfg.agent.server_url)
        rows.append(Row("llama-server", ok, body))
        if check_audio and guide.speaker is not None and guide.speaker.enabled:
            t0 = time.perf_counter()
            guide.speaker.speak("ARGUS audio check.")
            idle = guide.speaker.wait_until_idle(timeout=15.0)
            rows.append(Row("Piper playback", idle, f"synth+play {time.perf_counter() - t0:.2f} s "
                                                    "(completion only; a human must confirm hearing it)"))
        elif check_audio and speak:
            rows.append(Row("Piper playback", False, "Piper voice failed to load"))
        elif check_audio:
            rows.append(Row("Piper playback", None, "skipped (--quiet)"))

        report = {"kind": "argus-doctor", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "config_file": getattr(cfg, "source", ""),
                  "config": {"stereo": [cfg.camera.stereo_width, cfg.camera.stereo_height,
                                        cfg.camera.stereo_fps],
                             "wide": [cfg.camera.wide_width, cfg.camera.wide_height,
                                      cfg.camera.wide_fps],
                             "pixel_format": cfg.camera.pixel_format},
                  "rows": [asdict(r) for r in rows],
                  "all_ok": all(r.ok is not False for r in rows)}
        print_rows(rows)
        fails = [r.name for r in rows if r.ok is False]
        if fails:
            guide.failed(f"{len(fails)} checks failed: " + "; ".join(fails[:4]) + ".")
        else:
            guide.done("All checks passed.")
        return report
    finally:
        guide.close()


def print_rows(rows: list[Row]) -> None:
    w = max(len(r.name) for r in rows) + 2
    print("\nARGUS doctor")
    print("-" * (w + 60))
    for r in rows:
        mark = {True: "PASS", False: "FAIL", None: "info"}[r.ok]
        print(f"{mark:5} {r.name:<{w}}{r.detail}")
    print("-" * (w + 60))


def write_json(report: dict, name: str) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / name
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"report: {path}")
    return path


# ---------------------------------------------------------------------------
# --snap: worn-orientation snapshots
# ---------------------------------------------------------------------------
def run_snap(cfg: ArgusConfig, seconds: float = 10.0, count: int = 3, speak: bool = True) -> list[str]:
    from .guide import Guide
    guide = Guide(cfg.speech, "ARGUS snapshot", speak=speak, show=True)
    roles = assign_roles(capture_nodes(), cfg)
    if not roles:
        guide.failed("No cameras found.")
        guide.close()
        return []
    probes = _open_probes(cfg, roles)
    paths: list[str] = []
    try:
        guide.announce("Wear the rig exactly as you normally would, and face the room. "
                       f"Keep still for {int(seconds)} seconds.")
        _wait_frames(probes)
        t0 = time.monotonic()
        shots = 0
        while True:
            el = time.monotonic() - t0
            if el >= seconds:
                break
            remaining = int(seconds - el) + 1
            guide.say(f"{remaining}", force=False)
            due = int(el / (seconds / count))
            if due >= shots and shots < count and el > 1.0:
                paths += save_snaps(probes, tag=f"_{shots}")
                shots += 1
            frame = probes[next(iter(probes))].snapshot()
            guide.show(None if frame is None else cv2.resize(frame, (1920, 1080)),
                       sub=f"capturing {shots} of {count}", progress=el / seconds, delay_ms=30)
        paths += save_snaps(probes)
        guide.done(f"Captured {len(paths)} pictures.")
    finally:
        for p in probes.values():
            p.close()
        guide.close()
    print("\n".join(paths))
    return paths


# ---------------------------------------------------------------------------
# --skew-test: true exposure offset from the on-screen timecode
# ---------------------------------------------------------------------------
class TimecodeSampler:
    """Decodes each camera's newest frame on its own thread and keeps only
    (arrival time, decoded ms). Never buffers images: holding raw 960x600
    frames for a 20 s window at 60 fps exhausted the 8 GB board (OOM kill,
    2026-10-02)."""

    def __init__(self, probe: StreamProbe):
        self.probe = probe
        self.samples: list[tuple[float, int]] = []
        self.attempts = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"tc-{probe.role}")
        self._thread.start()

    def _run(self):
        from . import timecode as tc
        last = -1
        while not self._stop.is_set():
            frame, ts, seq = self.probe.newest()
            if frame is None or seq == last:
                time.sleep(0.002)
                continue
            last = seq
            self.attempts += 1
            v = tc.decode_frame(frame)
            if v is not None:
                self.samples.append((ts, v))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


def run_skew_test(cfg: ArgusConfig, seconds: float = 20.0, stereo_fps: int | None = None,
                  speak: bool = True) -> dict:
    from . import timecode as tc
    from .guide import Guide
    guide = Guide(cfg.speech, "ARGUS skew test", speak=speak, show=False)
    roles = assign_roles(capture_nodes(), cfg)
    if "left" not in roles or "right" not in roles:
        guide.failed("The skew test needs both stereo cameras plugged in.")
        guide.close()
        return {"error": "need both stereo cameras"}
    probes = _open_probes(cfg, {k: roles[k] for k in ("left", "right")},
                          stereo_fps=stereo_fps)
    win = "ARGUS skew test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    clock0 = time.perf_counter()

    def now_ms() -> int:
        return int((time.perf_counter() - clock0) * 1000)

    try:
        guide.announce("Skew test. Point both stereo cameras at the monitor, about half a metre "
                       "away, so the whole black and white pattern is visible. Hold still.")
        _wait_frames(probes)
        # Aim phase: show the pattern until both cameras decode it.
        aim_end = time.monotonic() + 60
        while time.monotonic() < aim_end:
            cv2.imshow(win, tc.render(now_ms()))
            cv2.waitKey(1)
            dl = probes["left"].latest
            dr = probes["right"].latest
            okl = dl is not None and tc.decode_frame(dl) is not None
            okr = dr is not None and tc.decode_frame(dr) is not None
            if okl and okr:
                break
            missing = " and ".join(r for r, ok in (("left", okl), ("right", okr)) if not ok)
            guide.say(f"The {missing} camera cannot read the pattern. Move closer or re-aim.")
        else:
            guide.failed("Both cameras never saw the pattern within one minute.")
            return {"error": "aim timeout"}
        guide.say(f"Good. Hold still for {int(seconds)} seconds.", force=True)
        for p in probes.values():
            p.reset_stats()
        samplers = {r: TimecodeSampler(p) for r, p in probes.items()}
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            cv2.imshow(win, tc.render(now_ms()))
            cv2.waitKey(1)
            remaining = int(end - time.monotonic()) + 1
            if remaining % 5 == 0:
                guide.say(f"{remaining} seconds.")
        for sm in samplers.values():
            sm.stop()
        guide.say("Measuring.", force=True)
        decoded = {r: [(t, v, int((t - clock0) * 1000)) for t, v in sm.samples]
                   for r, sm in samplers.items()}
        L, R = decoded["left"], decoded["right"]
        if len(L) < 20 or len(R) < 20:
            guide.failed(f"Too few readable frames: left {len(L)}, right {len(R)}.")
            return {"error": "too few decoded frames", "left": len(L), "right": len(R)}
        r_arr = np.array([x[0] for x in R])
        true_off, arr_off = [], []
        for t_l, v_l, _ in L:
            j = int(np.argmin(np.abs(r_arr - t_l)))
            t_r, v_r, _ = R[j]
            true_off.append(tc.wrap_diff_ms(v_l, v_r))       # left exposed later by this
            arr_off.append((t_l - t_r) * 1000.0)
        true_off = np.array(true_off, float)
        arr_off = np.array(arr_off, float)
        lat = {role: [((x[2] - x[1]) % 65536) for x in items] for role, items in decoded.items()}
        report = {
            "kind": "argus-skew-test", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stereo_fps": stereo_fps or cfg.camera.stereo_fps,
            "frames": {r: len(p.arrivals) for r, p in probes.items()},
            "decode_attempts": {r: sm.attempts for r, sm in samplers.items()},
            "decoded": {r: len(v) for r, v in decoded.items()},
            "true_offset_ms": {"mean": float(true_off.mean()), "median": float(np.median(true_off)),
                               "p95_abs": float(np.percentile(np.abs(true_off), 95))},
            "arrival_offset_ms": {"mean": float(arr_off.mean()), "median": float(np.median(arr_off)),
                                  "p95_abs": float(np.percentile(np.abs(arr_off), 95))},
            "arrival_minus_true_ms": {"mean": float((arr_off - true_off).mean()),
                                      "std": float((arr_off - true_off).std())},
            "display_to_arrival_latency_ms": {r: float(np.median(v)) for r, v in lat.items()},
            "note": "true offset is quantised by the monitor refresh; the mean over many pairs "
                    "is the estimate. Positive = left frame shows a later time than right.",
        }
        print(json.dumps(report, indent=2))
        guide.done(f"True offset about {abs(report['true_offset_ms']['mean']):.0f} milliseconds; "
                   f"arrival estimate {abs(report['arrival_offset_ms']['mean']):.0f}.")
        return report
    finally:
        for p in probes.values():
            p.close()
        cv2.destroyWindow(win)
        guide.close()


# ---------------------------------------------------------------------------
# --bandwidth: stereo fps with the wide camera closed vs streaming
# ---------------------------------------------------------------------------
def run_bandwidth(cfg: ArgusConfig, seconds: float = 8.0, rates=(30, 60, 80),
                  speak: bool = True) -> dict:
    from .guide import Guide
    guide = Guide(cfg.speech, "ARGUS bandwidth", speak=speak, show=False)
    roles = assign_roles(capture_nodes(), cfg)
    results = []
    try:
        guide.say("Bandwidth test. This takes about a minute; nothing to do.", force=True)
        for fps in rates:
            for wide_on in (False, True):
                if wide_on and "wide" not in roles:
                    continue
                probes = _open_probes(cfg, roles, stereo_fps=fps, wide=wide_on)
                try:
                    _wait_frames(probes)
                    time.sleep(1.0)
                    for p in probes.values():
                        p.reset_stats()
                    time.sleep(seconds)
                    entry = {"stereo_fps_requested": fps, "wide_streaming": wide_on,
                             "fps": {r: round(p.delivered_fps(), 2) for r, p in probes.items()},
                             "max_gap_ms": {r: round(p.max_gap_ms(), 1) for r, p in probes.items()}}
                    if "left" in probes and "right" in probes:
                        entry["skew"] = skew_stats(pair_skews_ms(probes["left"].arrivals,
                                                                 probes["right"].arrivals),
                                                   cfg.camera.max_skew_ms)
                    results.append(entry)
                    print(json.dumps(entry))
                finally:
                    for p in probes.values():
                        p.close()
                    time.sleep(0.5)
        wide_open = []
        if "wide" in roles:
            for _ in range(5):
                p = StreamProbe("wide", roles["wide"], cfg.camera.wide_width, cfg.camera.wide_height,
                                cfg.camera.wide_fps, cfg.camera.pixel_format)
                end = time.monotonic() + 10
                while p.first_frame_s is None and time.monotonic() < end:
                    time.sleep(0.01)
                wide_open.append(p.first_frame_s)
                p.close()
                time.sleep(0.5)
        report = {"kind": "argus-usb-bandwidth", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "roles": {r: {"port": n.usb_port, "link_mbps": n.link_mbps} for r, n in roles.items()},
                  "matrix": results, "wide_open_to_first_frame_s": wide_open}
        guide.done("Bandwidth test finished.")
        return report
    finally:
        guide.close()

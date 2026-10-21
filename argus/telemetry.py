"""Process-wide telemetry bus for the on-monitor dashboard.

Every loop publishes what it sees and decides here; the dashboard only reads.
Publishing is cheap (a lock and a reference swap) so the fast safety loop can
call it every tick. Nothing in this module influences behaviour: it is an
observation channel, never a control path.
"""
from __future__ import annotations

import collections
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    when: float
    kind: str      # "user" | "argus" | "safety" | "system"
    text: str


@dataclass
class Snapshot:
    frames: dict[str, np.ndarray]
    values: dict
    events: list[Event]
    detections: list
    generated: float = field(default_factory=time.monotonic)


class Telemetry:
    def __init__(self, max_events: int = 60):
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._values: dict = {}
        self._events: collections.deque[Event] = collections.deque(maxlen=max_events)
        self._detections: list = []
        self._detections_at = 0.0
        self.sinks: list = []          # callables(dict) — e.g. SessionLogger

    # ------------------------------------------------------------ publish
    def frame(self, name: str, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        with self._lock:
            self._frames[name] = frame

    def set(self, **values) -> None:
        from .sensitive import scrub
        with self._lock:
            self._values.update(scrub(values))

    def log(self, kind: str, text: str) -> None:
        if not text:
            return
        from .sensitive import mask_text
        text = mask_text(text)
        with self._lock:
            self._events.append(Event(time.monotonic(), kind, text))
        print(f"[{kind}] {text}", flush=True)
        self.record("say" if kind in ("argus", "safety") else kind, text=text,
                    channel=kind)

    def record(self, event: str, **fields) -> None:
        """Structured study event for every attached sink (never raises)."""
        if not self.sinks:
            return
        from .sensitive import scrub
        item = scrub({"t": time.monotonic(), "wall": time.time(), "event": event, **fields})
        for sink in list(self.sinks):
            try:
                sink(item)
            except Exception:  # noqa: BLE001 — logging must never break a loop
                pass

    def detections(self, dets: list) -> None:
        with self._lock:
            self._detections = list(dets)
            self._detections_at = time.monotonic()

    # ------------------------------------------------------------ read
    def snapshot(self) -> Snapshot:
        with self._lock:
            dets = self._detections if time.monotonic() - self._detections_at < 8.0 else []
            return Snapshot(dict(self._frames), dict(self._values), list(self._events), dets)

    def get(self, key: str, default=None):
        with self._lock:
            return self._values.get(key, default)


class SessionLogger:
    """Append-only JSONL study log: one event per line, flushed immediately
    so a crash loses nothing. Sensitive OCR text is masked before it is
    written (argus.sensitive)."""

    def __init__(self, path):
        import json
        from pathlib import Path
        self._json = json
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)
        self._lock = threading.Lock()

    def __call__(self, item: dict) -> None:
        from .sensitive import scrub
        item = scrub(item)
        line = self._json.dumps(item, default=float)
        with self._lock:
            self._f.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._f.close()


# One bus per process. Modules import this instead of threading a reference
# through every constructor.
bus = Telemetry()


class TegraStats(threading.Thread):
    """Sample RAM, swap and GR3D from tegrastats into the bus."""

    def __init__(self, telemetry: Telemetry = bus, interval_ms: int = 500):
        super().__init__(name="tegrastats", daemon=True)
        self.telemetry = telemetry
        self.interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

    def run(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            self.telemetry.set(**parse_tegrastats(line))

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()


def parse_tegrastats(line: str) -> dict:
    """Pull RAM/SWAP/GR3D/temperature fields out of one tegrastats line."""
    import re
    out: dict = {}
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if m:
        out["ram_used_mb"], out["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"SWAP (\d+)/(\d+)MB", line)
    if m:
        out["swap_used_mb"], out["swap_total_mb"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        out["gr3d_percent"] = int(m.group(1))
    m = re.search(r"gpu@([\d.]+)C", line)
    if m:
        out["gpu_temp_c"] = float(m.group(1))
    m = re.search(r"CPU \[([^\]]+)\]", line)
    if m:
        loads = [int(p.split("%")[0]) for p in m.group(1).split(",") if "%" in p]
        if loads:
            out["cpu_percent"] = sum(loads) / len(loads)
    return out

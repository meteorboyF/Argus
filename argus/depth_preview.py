"""Lossy display publication, isolated from safety's depth heartbeat.

Only the newest pending depth is retained. A slow renderer can drop previews;
it cannot delay safety or manufacture a fresh depth heartbeat.
"""
from __future__ import annotations
import threading
import time

import numpy as np

from .telemetry import bus


class LatestWorker:
    def __init__(self, consume):
        self._consume = consume
        self._condition = threading.Condition()
        self._pending = None
        self._halt = False
        self._thread = threading.Thread(target=self._run, daemon=True, name='depth-preview')
        self._thread.start()

    def submit(self, item):
        with self._condition:
            self._pending = item
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._halt or self._pending is not None)
                if self._halt:
                    return
                item, self._pending = self._pending, None
            try:
                self._consume(item)
            except Exception as exc:  # observation must never stop safety
                bus.record('depth_preview_error', error=str(exc))

    def stop(self):
        with self._condition:
            self._halt = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=2)


class DepthPreview(LatestWorker):
    def __init__(self, depth, rig):
        self.depth, self.rig = depth, rig
        self._last_links = 0.0
        super().__init__(self.render)

    def render(self, depth_m):
        from .dashboard import depth_to_color, disparity_to_color
        small = depth_m[::2, ::2]
        if self.depth.calibrated:
            color = depth_to_color(small)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                disp = (self.depth.cfg.focal_px*self.depth.cfg.baseline_m)/small
            disp[~np.isfinite(small)] = 0
            color = disparity_to_color(disp)
        bus.frame('depth_color', color)
        bus.set(fps_left=self.rig.left.fps, fps_right=self.rig.right.fps,
                fps_wide=self.rig.wide.fps)
        # USB sysfs discovery is display telemetry, sampled once per second.
        now = time.monotonic()
        if now-self._last_links >= 1:
            self._last_links = now
            bus.set(link_left=self.rig.left.link_mbps,
                    link_right=self.rig.right.link_mbps,
                    link_wide=self.rig.wide.link_mbps)
        wide, _, _ = self.rig.wide.latest()
        bus.frame('wide', wide)

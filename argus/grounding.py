"""YOLO-World open-vocabulary grounding — the agent's find_object tool.

The agent asks for an object by name (e.g. "keys"); this returns its bounding
box. We use the Ultralytics YOLOWorld model (which can run the .pt directly on
the Jetson GPU). A TensorRT engine path is the production target for ~160 ms
latency; until that's wired through ultralytics, the .pt runs via torch and is
still interactive on the Orin Nano.

set_classes() is the open-vocabulary mechanism: classes are provided at call
time, no retraining.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import GroundingConfig


@dataclass
class Detection:
    name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    center: tuple[int, int]


class Grounder:
    def __init__(self, cfg: GroundingConfig):
        self.cfg = cfg
        self._model = None
        self._load()

    def _load(self):
        from ultralytics import YOLOWorld
        weights = self.cfg.weights_pt if os.path.exists(self.cfg.weights_pt) else "yolov8s-worldv2.pt"
        self._model = YOLOWorld(weights)

    def find_object(self, name: str, frame_bgr: np.ndarray) -> Detection | None:
        """Detect the single best instance of `name` in the frame, or None."""
        self._model.set_classes([name])
        results = self._model.predict(
            frame_bgr, conf=self.cfg.conf_threshold, imgsz=self.cfg.imgsz, verbose=False
        )
        r = results[0]
        if len(r.boxes) == 0:
            return None
        # Highest-confidence box.
        best = max(r.boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = (int(v) for v in best.xyxy[0].tolist())
        return Detection(
            name=name,
            confidence=float(best.conf),
            bbox=(x1, y1, x2, y2),
            center=((x1 + x2) // 2, (y1 + y2) // 2),
        )

    def find_all(self, names: list[str], frame_bgr: np.ndarray) -> list[Detection]:
        """Detect any of several named classes (e.g. a hazard watchlist)."""
        self._model.set_classes(names)
        results = self._model.predict(
            frame_bgr, conf=self.cfg.conf_threshold, imgsz=self.cfg.imgsz, verbose=False
        )
        r = results[0]
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            dets.append(Detection(
                name=names[int(b.cls)],
                confidence=float(b.conf),
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
            ))
        return dets

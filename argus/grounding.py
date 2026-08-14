"""YOLO-World open-vocabulary grounding — the agent's find_object tool.

The agent asks for an object by name (e.g. "keys"); this returns its bounding
box. Production requires a verified TensorRT implementation. The existing engine
may contain a baked vocabulary and is deliberately refused until Feature 2
establishes its real binding/text-prompt contract.

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
        self._classes: list[str] | None = None
        self._load()

    def _load(self):
        if self.cfg.backend == "trt":
            if not os.path.exists(self.cfg.engine):
                raise RuntimeError(
                    f"Production YOLO-World TensorRT engine is missing: {self.cfg.engine}")
            raise RuntimeError(
                "YOLO-World TensorRT engine exists but its open-vocabulary contract "
                "is not verified and the runtime path is not implemented. Production "
                "startup is refused instead of silently using PyTorch.")
        if self.cfg.backend != "torch":
            raise ValueError(f"Unknown grounding backend: {self.cfg.backend!r}")
        if not self.cfg.allow_torch_fallback:
            raise RuntimeError(
                "Ultralytics PyTorch grounding is diagnostic-only and is disabled "
                "for production.")
        from ultralytics import YOLOWorld
        weights = self.cfg.weights_pt if os.path.exists(self.cfg.weights_pt) else "yolov8s-worldv2.pt"
        self._model = YOLOWorld(weights)

    def _set_classes(self, names: list[str]):
        # set_classes runs the CLIP text encoder — expensive on the Jetson.
        # Skip it when the vocabulary hasn't changed since the last call.
        if names != self._classes:
            self._model.set_classes(names)
            self._classes = names

    def find_object(self, name: str, frame_bgr: np.ndarray) -> Detection | None:
        """Detect the single best instance of `name` in the frame, or None."""
        self._set_classes([name])
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
        self._set_classes(names)
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

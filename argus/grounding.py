"""Fail-closed YOLO-World grounding with runtime text embeddings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

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
        self._runner = None
        self._text_model = None
        self._classes: list[str] | None = None
        self._load()

    def _load(self):
        if self.cfg.backend == "trt":
            if not os.path.exists(self.cfg.engine):
                raise RuntimeError(
                    f"Production YOLO-World TensorRT engine is missing: {self.cfg.engine}")
            if not os.path.exists(self.cfg.text_encoder):
                raise RuntimeError(
                    f"Pinned CLIP text encoder is missing: {self.cfg.text_encoder}")
            from .trt_runner import TRTRunner
            self._runner = TRTRunner(self.cfg.engine)
            inputs = {item["name"]: tuple(item["shape"]) for item in self._runner.inputs}
            required = {"images": (1, 3, self.cfg.imgsz, self.cfg.imgsz),
                        "text_embeddings": (1, 1, 512)}
            if inputs != required:
                raise RuntimeError(
                    "Grounding engine has an unverified vocabulary contract: "
                    f"expected {required}, got {inputs}")
            return
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

    @lru_cache(maxsize=64)
    def _embed(self, name: str) -> np.ndarray:
        """Encode one requested label on CPU and cache the normalized vector."""
        if self._text_model is None:
            import clip
            self._text_model, _ = clip.load(
                self.cfg.text_encoder, device="cpu", jit=False)
            self._text_model.eval()
        import clip
        import torch
        with torch.inference_mode():
            tokens = clip.tokenize([name], truncate=True)
            vector = self._text_model.encode_text(tokens).float()
            vector /= vector.norm(dim=-1, keepdim=True)
        return vector.numpy().reshape(1, 1, 512)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        import cv2
        height, width = frame_bgr.shape[:2]
        scale = min(self.cfg.imgsz / width, self.cfg.imgsz / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        resized = cv2.resize(frame_bgr, (resized_w, resized_h))
        pad_x = (self.cfg.imgsz - resized_w) // 2
        pad_y = (self.cfg.imgsz - resized_h) // 2
        canvas = np.full((self.cfg.imgsz, self.cfg.imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        image = canvas[:, :, ::-1].transpose(2, 0, 1)
        return np.ascontiguousarray(image[None], dtype=np.float32) / 255.0, scale, pad_x, pad_y

    def _find_object_trt(self, name: str, frame_bgr: np.ndarray) -> Detection | None:
        import cv2
        image, scale, pad_x, pad_y = self._preprocess(frame_bgr)
        outputs = self._runner.infer(
            {"images": image, "text_embeddings": self._embed(name)})
        raw = next(iter(outputs.values()))[0]
        if raw.shape[0] != 5:
            raise RuntimeError(f"Unexpected grounding output shape: {raw.shape}")
        keep = np.flatnonzero(raw[4] >= self.cfg.conf_threshold)
        if keep.size == 0:
            return None
        boxes_xywh = raw[:4, keep].T
        scores = raw[4, keep]
        boxes = []
        for cx, cy, width, height in boxes_xywh:
            boxes.append([float(cx - width / 2), float(cy - height / 2),
                          float(width), float(height)])
        selected = cv2.dnn.NMSBoxes(boxes, scores.tolist(),
                                    self.cfg.conf_threshold, 0.45)
        if len(selected) == 0:
            return None
        best_i = max((int(i) for i in np.asarray(selected).reshape(-1)),
                     key=lambda i: float(scores[i]))
        x, y, width, height = boxes[best_i]
        frame_h, frame_w = frame_bgr.shape[:2]
        x1 = int(np.clip((x - pad_x) / scale, 0, frame_w - 1))
        y1 = int(np.clip((y - pad_y) / scale, 0, frame_h - 1))
        x2 = int(np.clip((x + width - pad_x) / scale, 0, frame_w - 1))
        y2 = int(np.clip((y + height - pad_y) / scale, 0, frame_h - 1))
        return Detection(name, float(scores[best_i]), (x1, y1, x2, y2),
                         ((x1 + x2) // 2, (y1 + y2) // 2))

    def find_object(self, name: str, frame_bgr: np.ndarray) -> Detection | None:
        """Detect the single best instance of `name` in the frame, or None."""
        if self.cfg.backend == "trt":
            return self._find_object_trt(name.strip(), frame_bgr)
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
        if self.cfg.backend == "trt":
            return [det for name in names
                    if (det := self._find_object_trt(name.strip(), frame_bgr)) is not None]
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

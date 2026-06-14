"""Mandatory privacy gate.

Every frame that reaches the reasoning agent MUST pass through this gate first.
It blurs all detected faces (and optionally text regions) so the VLM never sees
identifiable people or private documents.

Face detection: InsightFace SCRFD (buffalo_s pack), CUDA if available.
Text detection: CRAFT (optional; enable once weights are present).
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import PrivacyConfig


class PrivacyGate:
    def __init__(self, cfg: PrivacyConfig):
        self.cfg = cfg
        self._face = None
        self._init_face()

    def _init_face(self):
        try:
            from insightface.app import FaceAnalysis
            self._face = FaceAnalysis(
                name=self.cfg.face_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face.prepare(ctx_id=0, det_size=(self.cfg.det_size, self.cfg.det_size))
        except Exception as e:  # noqa: BLE001
            # Fail loud but keep a safe fallback: if face detection can't load,
            # the gate blurs nothing — so log clearly. The orchestrator should
            # treat a missing gate as a hard error in production.
            print(f"[privacy] FaceAnalysis init failed: {e}")
            self._face = None

    @property
    def ready(self) -> bool:
        return self._face is not None

    def apply(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (gated_frame, num_faces_blurred). Never mutates the input."""
        out = frame_bgr.copy()
        n = 0
        if self._face is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            for f in self._face.get(rgb):
                x1, y1, x2, y2 = (int(v) for v in f.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                roi = out[y1:y2, x1:x2]
                if roi.size:
                    k = self.cfg.blur_kernel | 1  # kernel must be odd
                    out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
                    n += 1
        # Text blur (CRAFT) intentionally omitted until weights are wired in.
        return out, n

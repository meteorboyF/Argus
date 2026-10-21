"""VPI semi-global matching on the CUDA backend (Orin Nano has no PVA/OFA).

Selectable with depth.backend: vpi_sgm. Measured 2026-10-21 on live pairs
(reports/depth-backends-2026-10-21.json): 13.0 ms median at 480x300 with 64
disparities, vs 7.6 ms for the CUDA SAD matcher; denser output. Pixels whose
SGM confidence is below the threshold are returned invalid (-1), so the
safety rules treat them as UNKNOWN, never as free space.
"""
from __future__ import annotations

import numpy as np


class VpiSgmMatcher:
    def __init__(self, conf_threshold: int = 32767):
        import vpi
        self._vpi = vpi
        self.conf_threshold = conf_threshold
        # Smoke-run once so a broken VPI install fails at startup, not mid-walk.
        z = np.zeros((64, 128), np.uint8)
        self.compute(z, z, 16)

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray,
                max_disparity: int) -> np.ndarray:
        vpi = self._vpi
        h, w = left_gray.shape
        with vpi.Backend.CUDA:
            conf = vpi.Image((w, h), vpi.Format.U16)
            disp = vpi.stereodisp(vpi.asimage(left_gray), vpi.asimage(right_gray),
                                  out_confmap=conf, window=5, maxdisp=int(max_disparity),
                                  confthreshold=self.conf_threshold, includediagonals=False)
            d = disp.convert(vpi.Format.S16, backend=vpi.Backend.CUDA).cpu()
            c = conf.cpu()
        out = d.astype(np.float32) / 32.0          # Q10.5 fixed point
        out[(c < self.conf_threshold) | (out <= 0)] = -1.0
        return out

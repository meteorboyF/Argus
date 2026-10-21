#!/usr/bin/env python3
"""Stereo backend spike (M4): CUDA SAD (production) vs VPI SGM (CUDA).

Grabs live left/right pairs from the rig, converts to grey, and times each
backend on the same frames at the given resolutions and disparity ranges.
Reports latency median/p95, valid-pixel fraction, and tegrastats GR3D.

Pairs are NOT rectified when no calibration exists: latency and GPU load are
valid measurements, valid-pixel % is only indicative until calibration.

    python3 scripts/bench_depth.py --frames 40 --out reports/depth-backends-2026-10-21.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class Tegra:
    def __init__(self):
        self.samples: list[tuple[float, int]] = []
        self.p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for line in self.p.stdout:
            m = re.search(r"GR3D_FREQ (\d+)%", line)
            if m:
                self.samples.append((time.monotonic(), int(m.group(1))))

    def window(self, t0, t1):
        g = [v for t, v in self.samples if t0 <= t <= t1]
        return {"gr3d_peak": max(g) if g else None,
                "gr3d_mean": round(float(np.mean(g)), 1) if g else None}

    def stop(self):
        self.p.terminate()


def grab_pairs(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from argus.config import load_config
    from argus.doctor import _open_probes, _wait_frames, assign_roles, capture_nodes
    cfg = load_config()
    roles = assign_roles(capture_nodes(), cfg)
    probes = _open_probes(cfg, {k: roles[k] for k in ("left", "right")})
    _wait_frames(probes)
    time.sleep(0.5)
    pairs = []
    try:
        while len(pairs) < n:
            l, r = probes["left"].snapshot(0.5), probes["right"].snapshot(0.5)
            if l is not None and r is not None:
                pairs.append((cv2.cvtColor(l, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)))
            time.sleep(0.05)
    finally:
        for p in probes.values():
            p.close()
    return pairs


def run_sad(pairs, size, maxdisp):
    from argus.cuda_stereo import CudaStereoMatcher
    m = CudaStereoMatcher()
    out = []
    for l, r in pairs:
        l2, r2 = cv2.resize(l, size), cv2.resize(r, size)
        t = time.perf_counter()
        d = m.compute(l2, r2, maxdisp, 2, 10)
        out.append((time.perf_counter() - t, d))
    return out


def run_vpi(pairs, size, maxdisp, conf_threshold=32767):
    import vpi
    out = []
    with vpi.Backend.CUDA:
        for l, r in pairs:
            l2, r2 = cv2.resize(l, size), cv2.resize(r, size)
            t = time.perf_counter()
            vl, vr = vpi.asimage(l2), vpi.asimage(r2)
            conf = vpi.Image(size, vpi.Format.U16)
            disp = vpi.stereodisp(vl, vr, out_confmap=conf, window=5, maxdisp=maxdisp,
                                  confthreshold=conf_threshold, includediagonals=False)
            disp_np = disp.convert(vpi.Format.S16, backend=vpi.Backend.CUDA).cpu().astype(np.float32) / 32.0
            conf_np = conf.cpu()
            dt = time.perf_counter() - t
            disp_np[conf_np < conf_threshold] = -1
            out.append((dt, disp_np))
    return out


def summarise(name, res, tegra, t0, t1):
    lat = np.array([r[0] for r in res[2:]]) * 1000          # drop warm-up
    valid = [float(np.mean(np.asarray(r[1]) > 0)) for r in res[2:]]
    s = {"backend": name, "median_ms": round(float(np.median(lat)), 2),
         "p95_ms": round(float(np.percentile(lat, 95)), 2),
         "valid_fraction": round(float(np.mean(valid)), 3)}
    s.update(tegra.window(t0, t1))
    print(json.dumps(s), flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = grab_pairs(args.frames)
    tegra = Tegra()
    time.sleep(0.5)
    results = []
    configs = [("cuda_sad", (480, 300), 64), ("cuda_sad", (480, 300), 128),
               ("vpi_sgm", (480, 300), 64), ("vpi_sgm", (480, 300), 128),
               ("vpi_sgm", (960, 600), 128), ("vpi_sgm", (960, 600), 256)]
    try:
        for name, size, md in configs:
            t0 = time.monotonic()
            res = (run_sad if name == "cuda_sad" else run_vpi)(pairs, size, md)
            t1 = time.monotonic()
            s = summarise(f"{name} {size[0]}x{size[1]} maxdisp {md}", res, tegra, t0, t1)
            results.append(s)
            # Save one visual per config for inspection.
            d = np.asarray(res[len(res) // 2][1], np.float32)
            vis = cv2.applyColorMap(np.clip(d / md * 255, 0, 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
            vis[d <= 0] = 0
            cv2.imwrite(f"/tmp/argus_snap/disp_{name}_{size[0]}_{md}.jpg", vis)
            time.sleep(0.5)
    finally:
        tegra.stop()
    rep = {"kind": "argus-depth-backends", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "frames": len(pairs), "rectified": False,
           "note": "uncalibrated: latency/GR3D valid; valid_fraction only indicative",
           "results": results}
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

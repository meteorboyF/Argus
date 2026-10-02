#!/usr/bin/env python3
"""Measured VLM variants for ARGUS (M3): latency, peak RAM, GR3D, answers.

For each variant this starts llama-server with the given flags, waits for
/health, runs every task (image + prompt) `--reps` times, records server-side
timings, the server's peak RSS (VmHWM), the drop in system MemAvailable,
tegrastats GR3D, and the answers, then stops the server. Every image passes
through the InsightFace privacy gate first (fail-closed), exactly as in the
runtime.

    python3 scripts/bench_vlm.py --variants baseline,vision_mmproj --out reports/x.json

Task images are sized by token budget: one Gemma 4 vision token covers
48x48 px (16 px patches merged 3x3), so `tokens: 280` sends ~280 tokens of
pixels instead of the old 256 px thumbnail (~15 tokens, upscaled to 40).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BIN = "/opt/argus/llama.cpp/build/bin/llama-server"
M = "/opt/argus/models"
PORT = 8091  # never collide with a production server on 8080
URL = f"http://127.0.0.1:{PORT}"

COMMON = ["--device", "CUDA0", "-ngl", "99", "--parallel", "1", "--fit", "off",
          "--flash-attn", "on", "--reasoning", "off", "--cache-ram", "0",
          "--ctx-size", "2048", "--jinja", "--host", "127.0.0.1", "--port", str(PORT)]

VARIANTS: dict[str, dict] = {
    # Production as of 2026-09-29: full mmproj on CPU, 256 px client cap.
    "baseline": {"model": f"{M}/gemma-4-E2B-it-Q4_K_M.gguf",
                 "mmproj": f"{M}/mmproj-gemma4-e2b-f16.gguf",
                 "args": ["--no-mmproj-offload"], "max_side": 256},
    "vision_mmproj": {"model": f"{M}/gemma-4-E2B-it-Q4_K_M.gguf",
                      "mmproj": f"{M}/mmproj-gemma4-e2b-vision-f16.gguf",
                      "args": ["--no-mmproj-offload"], "max_side": 256},
    "vision_mmproj_gpu": {"model": f"{M}/gemma-4-E2B-it-Q4_K_M.gguf",
                          "mmproj": f"{M}/mmproj-gemma4-e2b-vision-f16.gguf",
                          "args": [], "max_side": 256},
    "vision_gpu_tokens": {"model": f"{M}/gemma-4-E2B-it-Q4_K_M.gguf",
                          "mmproj": f"{M}/mmproj-gemma4-e2b-vision-f16.gguf",
                          "args": ["--image-max-tokens", "560", "-ub", "1024", "-b", "1024"],
                          "tokens": True},
    "qat_vision_gpu_tokens": {"model": f"{M}/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
                              "mmproj": f"{M}/mmproj-gemma4-e2b-vision-f16.gguf",
                              "args": ["--image-max-tokens", "560", "-ub", "1024", "-b", "1024"],
                              "tokens": True},
}

FRAMES = REPO / "reports" / "bakeoff" / "frames"


def resize_to_tokens(img: np.ndarray, tokens: int, cell: int = 48) -> np.ndarray:
    """Scale so (w/cell)*(h/cell) ~= tokens, never upscaling past the source."""
    h, w = img.shape[:2]
    s = math.sqrt(tokens * cell * cell / (w * h))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(cell, int(w * s)), max(cell, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def load_tasks(path: Path) -> list[dict]:
    tasks = json.loads(path.read_text())
    from argus.config import PrivacyConfig
    from argus.privacy import PrivacyGate
    gate = PrivacyGate(PrivacyConfig())
    if not gate.ready:
        raise SystemExit("privacy gate unavailable: refusing to send images to a VLM")
    for t in tasks:
        img = cv2.imread(str(FRAMES / t["image"]))
        if img is None:
            raise SystemExit(f"missing frame {t['image']}")
        t["_img"], t["_faces"] = gate.apply(img)
    return tasks


class Tegra:
    """Background tegrastats sampler: GR3D % and RAM used (MB)."""

    def __init__(self):
        self.gr3d: list[int] = []
        self.ram: list[int] = []
        self.p = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        for line in self.p.stdout:
            g = re.search(r"GR3D_FREQ (\d+)%", line)
            r = re.search(r"RAM (\d+)/", line)
            if g:
                self.gr3d.append(int(g.group(1)))
            if r:
                self.ram.append(int(r.group(1)))

    def mark(self) -> tuple[int, int]:
        return len(self.gr3d), len(self.ram)

    def window(self, m: tuple[int, int]) -> dict:
        g, r = self.gr3d[m[0]:], self.ram[m[1]:]
        return {"gr3d_peak": max(g) if g else None,
                "gr3d_mean": round(float(np.mean(g)), 1) if g else None,
                "ram_peak_mb": max(r) if r else None}

    def stop(self):
        self.p.terminate()


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 0


def proc_status(pid: int) -> dict:
    out = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmHWM", "VmRSS", "RssAnon", "RssFile")):
                k, v = line.split(":")
                out[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return out


def start_server(v: dict, log: Path) -> subprocess.Popen:
    cmd = [BIN, "--model", v["model"], "--mmproj", v["mmproj"], *COMMON, *v["args"]]
    f = open(log, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    t0 = time.time()
    while time.time() - t0 < 240:
        if p.poll() is not None:
            raise RuntimeError(f"server exited rc={p.returncode}; see {log}")
        try:
            if requests.get(URL + "/health", timeout=1).status_code == 200:
                return p
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become healthy in 240 s")


def stop_server(p: subprocess.Popen):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=20)
    except Exception:  # noqa: BLE001
        os.killpg(p.pid, signal.SIGKILL)


def ask(img_url: str, prompt: str, max_tokens: int) -> dict:
    payload = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img_url}},
        {"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens, "temperature": 0.0,
        # Every real query brings a new frame; never let a repeated benchmark
        # image hit the KV prefix cache and report an unrealistically fast run.
        "cache_prompt": False}
    t0 = time.perf_counter()
    r = requests.post(URL + "/v1/chat/completions", json=payload, timeout=300)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    return {"wall_s": round(wall, 3), "text": j["choices"][0]["message"].get("content", ""),
            "timings": {k: j.get("timings", {}).get(k) for k in
                        ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second")}}


def run_variant(name: str, v: dict, tasks: list[dict], reps: int, tegra: Tegra,
                logdir: Path) -> dict:
    from argus.doctor import oc_events
    oc0 = oc_events()
    avail0 = mem_available_mb()
    m0 = tegra.mark()
    t_start = time.perf_counter()
    p = start_server(v, logdir / f"llama-{name}.log")
    load_s = time.perf_counter() - t_start
    res = {"variant": name, "model": Path(v["model"]).name, "mmproj": Path(v["mmproj"]).name,
           "args": v["args"], "load_s": round(load_s, 1), "tasks": []}
    try:
        idle = proc_status(p.pid)
        res["idle_rss_mb"] = idle.get("VmRSS")
        res["idle_mem_drop_mb"] = avail0 - mem_available_mb()
        min_avail = mem_available_mb()
        for t in tasks:
            if v.get("tokens"):
                img = resize_to_tokens(t["_img"], t.get("tokens", 280))
            else:
                img = resize_max_side(t["_img"], v.get("max_side", 256))
            url = data_url(img)
            runs = []
            for i in range(reps + 1):  # first run = cold for this image
                mk = tegra.mark()
                r = ask(url, t["prompt"], t.get("max_tokens", 64))
                r.update(tegra.window(mk))
                r["cold"] = i == 0
                runs.append(r)
                min_avail = min(min_avail, mem_available_mb())
            warm = [r["wall_s"] for r in runs if not r["cold"]]
            res["tasks"].append({
                "id": t["id"], "image": t["image"], "sent_px": [img.shape[1], img.shape[0]],
                "faces_blurred": t["_faces"],
                "cold_s": runs[0]["wall_s"],
                "warm_median_s": round(float(np.median(warm)), 3) if warm else None,
                "warm_p95_s": round(float(np.percentile(warm, 95)), 3) if warm else None,
                "prompt_tokens": runs[-1]["timings"]["prompt_n"],
                "answer": runs[-1]["text"], "expect": t.get("expect"), "runs": runs})
        st = proc_status(p.pid)
        res["peak_rss_mb"] = st.get("VmHWM")
        res["rss_mb"] = st.get("VmRSS")
        res["peak_mem_drop_mb"] = avail0 - min_avail
        res.update({k: v for k, v in tegra.window(m0).items()})
        oc1 = oc_events()
        res["oc_throttle_events"] = {k: oc1.get(k, 0) - oc0.get(k, 0) for k in oc1}
    finally:
        stop_server(p)
        time.sleep(3)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--tasks", default=str(FRAMES.parent / "tasks_m3.json"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = load_tasks(Path(args.tasks))
    logdir = Path("/tmp/argus_bench")
    logdir.mkdir(exist_ok=True)
    tegra = Tegra()
    report = {"kind": "argus-vlm-bench", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "llama_cpp": "ef8268feee28ae943958049bf3bbab4bda99c0ea", "reps": args.reps,
              "results": []}
    try:
        for name in args.variants.split(","):
            print(f"== {name}", flush=True)
            r = run_variant(name, VARIANTS[name], tasks, args.reps, tegra, logdir)
            report["results"].append(r)
            print(json.dumps({k: r.get(k) for k in ("variant", "load_s", "idle_rss_mb", "peak_rss_mb",
                                                    "peak_mem_drop_mb", "gr3d_peak")}), flush=True)
            for t in r["tasks"]:
                print(f"   {t['id']:10} {t['sent_px']} tok={t['prompt_tokens']} cold={t['cold_s']}s "
                      f"warm={t['warm_median_s']}s :: {t['answer'][:100]!r}", flush=True)
            Path(args.out).write_text(json.dumps(report, indent=2))
    finally:
        tegra.stop()
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Environment self-test for the Jetson bring-up.

Checks, in order: Python deps, CUDA/torch, model/engine files, cameras, and the
llama.cpp server. Prints a clear PASS/FAIL per item so you know exactly what's
left to do. Non-fatal items warn; fatal ones fail the run.
"""
from __future__ import annotations

import os
import shutil

from .config import ArgusConfig


def _ok(label, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return cond


def run_selftest(cfg: ArgusConfig) -> bool:
    print("ARGUS self-test\n" + "=" * 40)
    all_ok = True

    print("\n1. Python dependencies")
    for mod in ["cv2", "numpy", "requests", "ultralytics", "faster_whisper",
                "openwakeword", "onnxruntime"]:
        try:
            __import__(mod)
            _ok(mod, True)
        except Exception as e:  # noqa: BLE001
            all_ok &= _ok(mod, False, str(e))

    print("\n2. CUDA / Torch")
    try:
        import torch
        cu = torch.cuda.is_available()
        _ok("torch.cuda.is_available()", cu,
            torch.cuda.get_device_name(0) if cu else "no CUDA")
    except Exception as e:  # noqa: BLE001
        _ok("torch import", False, str(e))

    print("\n3. TensorRT (Jetson)")
    try:
        import tensorrt  # noqa: F401
        _ok("tensorrt import", True)
    except Exception as e:  # noqa: BLE001
        _ok("tensorrt import", False, f"{e} (engines will fall back / be unavailable)")

    print("\n4. Model & engine files")
    files = {
        "YOLO-World weights": cfg.grounding.weights_pt,
        "YOLO-World ONNX": cfg.grounding.onnx,
        "YOLO-World TRT engine": cfg.grounding.engine,
        "Gemma GGUF": cfg.agent.model_gguf,
        "Gemma mmproj": cfg.agent.mmproj_gguf,
        "Piper voice": cfg.speech.piper_voice,
    }
    for label, path in files.items():
        _ok(label, os.path.exists(path), path)

    print("\n5. Cameras")
    try:
        import cv2
        found = []
        for idx in (cfg.camera.left_index, cfg.camera.right_index, cfg.camera.wide_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                found.append(idx)
            cap.release()
        _ok("3 cameras open", len(found) == 3, f"opened indices {found}")
    except Exception as e:  # noqa: BLE001
        _ok("camera probe", False, str(e))

    print("\n6. llama.cpp server (Gemma 4)")
    try:
        import requests
        r = requests.get(cfg.agent.server_url.rstrip("/") + "/health", timeout=2)
        _ok("llama server /health", r.status_code == 200, cfg.agent.server_url)
    except Exception as e:  # noqa: BLE001
        _ok("llama server reachable", False,
            f"{e} (start it with scripts/run_llama_server.sh)")

    print("\n7. trtexec available")
    _ok("trtexec on PATH or /usr/src/tensorrt/bin",
        bool(shutil.which("trtexec")) or os.path.exists("/usr/src/tensorrt/bin/trtexec"))

    print("\n" + "=" * 40)
    print("Self-test complete. Fatal deps OK:" , all_ok)
    return all_ok

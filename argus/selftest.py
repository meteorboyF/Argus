"""Environment self-test for the Jetson bring-up.

Checks, in order: Python deps, CUDA/torch, TensorRT, model/engine files, stereo
calibration, camera discovery, audio devices, the privacy gate, and the
llama.cpp server. Prints a clear PASS/FAIL per item so you know exactly what's
left to do. Non-fatal items warn; fatal ones fail the run.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from .config import ArgusConfig


def _ok(label, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def run_selftest(cfg: ArgusConfig) -> bool:
    print("ARGUS self-test\n" + "=" * 40)
    all_ok = True

    print("\n1. Python dependencies")
    for mod in ["cv2", "numpy", "requests", "yaml", "ultralytics", "faster_whisper",
                "openwakeword", "onnxruntime", "insightface", "sounddevice", "piper"]:
        try:
            __import__(mod)
            _ok(mod, True)
        except Exception as e:  # noqa: BLE001
            fatal = mod in ("cv2", "numpy", "requests", "yaml")
            ok = _ok(mod, False, str(e))
            all_ok &= ok or not fatal

    print("\n2. CUDA / Torch")
    try:
        import torch
        cu = torch.cuda.is_available()
        all_ok &= _ok("torch.cuda.is_available()", cu,
            torch.cuda.get_device_name(0) if cu
            else "no CUDA — on Jetson install NVIDIA's torch wheel, not pip default")
    except Exception as e:  # noqa: BLE001
        all_ok &= _ok("torch import", False, str(e))

    print("\n3. TensorRT (Jetson)")
    try:
        import tensorrt  # noqa: F401
        all_ok &= _ok("tensorrt import", True)
    except Exception as e:  # noqa: BLE001
        all_ok &= _ok("tensorrt import", False, f"{e} (production cannot start)")
    all_ok &= _ok("trtexec on PATH or /usr/src/tensorrt/bin",
        bool(shutil.which("trtexec")) or os.path.exists("/usr/src/tensorrt/bin/trtexec"))
    try:
        import pycuda  # noqa: F401
        all_ok &= _ok("pycuda import", True)
    except Exception as e:  # noqa: BLE001
        all_ok &= _ok("pycuda import", False, str(e))

    print("\n4. Model & engine files")
    files = {
        "YOLO-World weights (provenance)": cfg.grounding.weights_pt,
        "YOLO-World ONNX (provenance)": cfg.grounding.onnx,
        "YOLO-World TRT engine (required)": cfg.grounding.engine,
        "RAFT-Stereo TRT engine (required)": cfg.depth.raft_engine,
        "Gemma GGUF (required)": cfg.agent.model_gguf,
        "Gemma mmproj (required)": cfg.agent.mmproj_gguf,
        "Piper voice (required for TTS)": cfg.speech.piper_voice,
    }
    for label, path in files.items():
        all_ok &= _ok(label, os.path.exists(path), path)

    all_ok &= _ok("production depth backend selected",
                  cfg.depth.backend == "raft_trt" and not cfg.depth.allow_cpu_fallback,
                  f"backend={cfg.depth.backend}, cpu_fallback={cfg.depth.allow_cpu_fallback}")
    all_ok &= _ok("production grounding backend selected",
                  cfg.grounding.backend == "trt" and not cfg.grounding.allow_torch_fallback,
                  f"backend={cfg.grounding.backend}, torch_fallback={cfg.grounding.allow_torch_fallback}")

    print("\n5. Stereo calibration")
    calib = cfg.depth.calibration_file
    if os.path.exists(calib):
        try:
            data = np.load(calib, allow_pickle=True)
            rms = float(data["rms"]) if "rms" in data else float("nan")
            verr = float(data["vertical_error"]) if "vertical_error" in data else float("nan")
            size = tuple(int(v) for v in data["image_size"])
            all_ok &= _ok("calibration file loads", True,
                f"{calib} (rms {rms:.2f} px, vert {verr:.2f} px, size {size[0]}x{size[1]})")
            all_ok &= _ok("calibration quality (rms < 1.5, vert < 2.0)",
                rms < 1.5 and verr < 2.0,
                "re-run scripts/calibrate_stereo.py" if not (rms < 1.5 and verr < 2.0) else "")
            from .cameras import transformed_size
            runtime_size = transformed_size(cfg.camera.stereo_width,
                                            cfg.camera.stereo_height,
                                            cfg.camera.left_rotation)
            all_ok &= _ok("calibration matches runtime resolution",
                size == runtime_size,
                f"calibrated {size}, runtime "
                f"{runtime_size} — frames will be "
                "resized (works, but re-calibrate at the runtime resolution for accuracy)")
            all_ok &= _ok("left/right USB ports recorded", "left_port" in data and str(data["left_port"]),
                "older calibration — re-run calibrate_stereo.py to bind physical cameras")
        except Exception as e:  # noqa: BLE001
            all_ok &= _ok("calibration file loads", False, str(e))
    else:
        all_ok &= _ok("calibration file", False,
            f"{calib} missing — production hazard speech is suppressed")

    print("\n6. Cameras (smart discovery)")
    try:
        from .cameras import discover_rig, probe_cameras
        nodes = probe_cameras(cfg.camera.max_probe_index, verbose=True)
        all_ok &= _ok("3 working cameras found", len(nodes) >= 3, f"found {len(nodes)}")
        if len(nodes) >= 3:
            li, ri, wi = discover_rig(cfg.camera)
            all_ok &= _ok("rig resolved (left/right/wide)", True, f"video{li}/video{ri}/video{wi}")
    except Exception as e:  # noqa: BLE001
        all_ok &= _ok("camera probe", False, str(e))

    print("\n7. Audio devices")
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        ins = [d["name"] for d in devs if d["max_input_channels"] > 0]
        outs = [d["name"] for d in devs if d["max_output_channels"] > 0]
        all_ok &= _ok("microphone present", len(ins) > 0, f"{len(ins)} input device(s)")
        all_ok &= _ok("speaker present", len(outs) > 0, f"{len(outs)} output device(s)")
    except Exception as e:  # noqa: BLE001
        all_ok &= _ok("audio probe", False, str(e))

    print("\n8. Privacy gate")
    try:
        from .privacy import PrivacyGate
        gate = PrivacyGate(cfg.privacy)
        ok = _ok("face detector initialises", gate.ready,
                 "" if gate.ready else "agent calls will be refused while require_gate=true")
        if cfg.privacy.require_gate:
            all_ok &= ok
    except Exception as e:  # noqa: BLE001
        _ok("privacy gate", False, str(e))
        all_ok = False

    print("\n9. llama.cpp server (Gemma)")
    try:
        import requests
        r = requests.get(cfg.agent.server_url.rstrip("/") + "/health", timeout=2)
        all_ok &= _ok("llama server /health", r.status_code == 200, cfg.agent.server_url)
    except Exception as e:  # noqa: BLE001
        all_ok &= _ok("llama server reachable", False,
            f"{e} (start it with scripts/run_llama_server.sh)")

    print("\n" + "=" * 40)
    print("Self-test complete. Fatal checks OK:", all_ok)
    return all_ok

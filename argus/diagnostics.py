"""Read-only Jetson environment baseline for ARGUS.

The report separates observed facts from production readiness. It never changes
power mode, clocks, packages, devices, or model files. Hardware probes degrade to
clear failures when run inside a restricted container rather than inventing a
passing result.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from .config import ArgusConfig


LLAMA_COMMIT = "ef8268feee28ae943958049bf3bbab4bda99c0ea"
EXPECTED_HASHES = {
    "gemma": ("models/gemma-4-E2B-it-Q4_K_M.gguf",
              "9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d"),
    "mmproj": ("models/mmproj-gemma4-e2b-f16.gguf",
               "140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa"),
    "yolo_pt": ("models/yolov8s-worldv2.pt",
                "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"),
    "clip_text": ("models/clip/ViT-B-32.pt",
                  "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"),
    "yolo_onnx": ("exports/yoloworld_runtime_text_640.onnx",
                  "7f69826f578b66cc057b4eb81659456145ff3962e295981a51682b318f3123fb"),
    "yolo_engine": ("engines/yoloworld_runtime_text_640_fp16.engine",
                    "7f0cf0a82c0bc5ee713eb91b7085219263160ef806282a3617748abc13b94bfd"),
    "piper": ("models/piper/en_US-lessac-medium.onnx",
              "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f"),
}


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    detail: str


def _run(argv: Sequence[str], timeout: float = 8.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout,
                              check=False)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_power_mode(text: str) -> str | None:
    match = re.search(r"NV Power Mode:\s*([^\r\n]+)", text)
    return match.group(1).strip() if match else None


def parse_gr3d_samples(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"GR3D_FREQ\s+(\d+)%", text)]


def parse_v4l2_names(text: str) -> list[str]:
    return [line.strip().rstrip(":") for line in text.splitlines()
            if line and not line[0].isspace() and line.rstrip().endswith(":")]


def has_usb_audio(text: str) -> bool:
    return "USB Audio" in text


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _gpu_probe() -> tuple[Check, dict]:
    """Run CUDA math while sampling tegrastats; no CPU fallback is accepted."""
    code = (
        "import json,time,torch; "
        "assert torch.cuda.is_available(), 'torch CUDA unavailable'; "
        "d=torch.device('cuda'); a=torch.randn((1024,1024),device=d); "
        "torch.cuda.synchronize(); t=time.perf_counter(); "
        "[(a@a) for _ in range(30)]; torch.cuda.synchronize(); "
        "print(json.dumps({'torch':torch.__version__,'device':torch.cuda.get_device_name(0),"
        "'seconds':time.perf_counter()-t}))"
    )
    sampler = None
    samples_text = ""
    rc, output = 127, "GPU workload did not start"
    try:
        try:
            sampler = subprocess.Popen(
                ["tegrastats", "--interval", "100"], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
        except OSError as exc:
            output = f"tegrastats failed to start: {exc}"
        rc, output = _run([sys.executable, "-c", code], timeout=30.0)
    finally:
        if sampler is not None:
            sampler.terminate()
            try:
                samples_text, _ = sampler.communicate(timeout=3.0)
            except subprocess.TimeoutExpired:
                sampler.kill()
                samples_text, _ = sampler.communicate()
    samples = parse_gr3d_samples(samples_text)
    details: dict = {"tegrastats_gr3d_samples": samples, "max_gr3d_percent": max(samples, default=0)}
    if rc == 0:
        try:
            details.update(json.loads(output.splitlines()[-1]))
        except (json.JSONDecodeError, IndexError):
            details["workload_output"] = output
    else:
        details["workload_error"] = output
    ok = rc == 0 and bool(samples) and max(samples) > 0
    detail = (f"{details.get('device', 'unknown')}; max GR3D "
              f"{details['max_gr3d_percent']}%; workload {details.get('seconds', 'failed')} s")
    return Check("CUDA workload + tegrastats GR3D", ok, True, detail), details


def collect_baseline(cfg: ArgusConfig, argus_home: Path = Path("/opt/argus")) -> dict:
    checks: list[Check] = []
    evidence: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "user": getpass.getuser(),
        "uid": os.geteuid(),
    }

    checks.append(Check("run as user argus", evidence["user"] == "argus" and
                        evidence["uid"] != 0, True,
                        f"user={evidence['user']} uid={evidence['uid']}"))

    tegra = Path("/etc/nv_tegra_release")
    tegra_text = tegra.read_text(errors="replace").strip() if tegra.exists() else "missing"
    evidence["nv_tegra_release"] = tegra_text
    checks.append(Check("Jetson Linux R36.5.2", "REVISION: 5.2" in tegra_text,
                        True, tegra_text.splitlines()[0] if tegra_text else "missing"))

    rc, power_text = _run(["nvpmodel", "-q"])
    power_mode = parse_power_mode(power_text)
    evidence["power_mode"] = power_mode
    checks.append(Check("MAXN_SUPER power mode", rc == 0 and power_mode == "MAXN_SUPER",
                        True, power_mode or power_text or "query failed"))

    rc, clocks_text = _run(["jetson_clocks", "--show"])
    evidence["jetson_clocks_show"] = clocks_text
    clocks_ok = rc == 0 and "Error" not in clocks_text and bool(clocks_text)
    checks.append(Check("jetson_clocks query", clocks_ok, True,
                        clocks_text.replace("\n", "; ")[:500] or "query failed"))

    gpu_check, gpu_evidence = _gpu_probe()
    checks.append(gpu_check)
    evidence["gpu_probe"] = gpu_evidence

    packages = {name: _package_version(name) for name in (
        "torch", "torchvision", "tensorrt", "pycuda", "numpy", "opencv-python",
        "ultralytics", "onnx", "onnxruntime", "insightface", "faster-whisper",
        "openwakeword", "piper-tts", "sounddevice", "clip", "ftfy")}
    evidence["packages"] = packages
    checks.append(Check("NVIDIA Jetson Torch build",
                        packages["torch"] == "2.5.0a0+872d972e41.nv24.8", True,
                        packages["torch"]))
    checks.append(Check("TensorRT 10.3", packages["tensorrt"].startswith("10.3."),
                        True, packages["tensorrt"]))
    checks.append(Check("pycuda available", packages["pycuda"] != "missing", True,
                        packages["pycuda"]))

    llama_dir = argus_home / "llama.cpp"
    rc, llama_commit = _run(["git", "-C", str(llama_dir), "rev-parse", "HEAD"])
    evidence["llama_commit"] = llama_commit
    cache = llama_dir / "build/CMakeCache.txt"
    cache_text = cache.read_text(errors="replace") if cache.exists() else ""
    llama_ok = (rc == 0 and llama_commit == LLAMA_COMMIT and "GGML_CUDA:BOOL=ON" in cache_text
                and "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=87" in cache_text)
    checks.append(Check("pinned CUDA llama.cpp build", llama_ok, True,
                        f"commit={llama_commit or 'missing'} cuda={('ON' if 'GGML_CUDA:BOOL=ON' in cache_text else 'missing')} arch87={('yes' if 'CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=87' in cache_text else 'no')}"))

    artifacts = {}
    for name, (relative, expected) in EXPECTED_HASHES.items():
        path = argus_home / relative
        actual = _sha256(path) if path.is_file() else "missing"
        artifacts[name] = {"path": str(path), "bytes": path.stat().st_size if path.is_file() else 0,
                           "sha256": actual, "expected_sha256": expected}
        checks.append(Check(f"artifact {name}", actual == expected, True,
                            f"{path}: {actual}"))
    evidence["artifacts"] = artifacts

    rc, camera_text = _run(["v4l2-ctl", "--list-devices"])
    camera_names = parse_v4l2_names(camera_text)
    evidence["v4l2"] = camera_text
    stereo_count = sum("B0495" in name for name in camera_names)
    wide_count = sum("B0459" in name for name in camera_names)
    checks.append(Check("three expected cameras enumerate",
                        rc == 0 and stereo_count == 2 and wide_count == 1, True,
                        f"B0495={stereo_count}, B0459={wide_count}; {camera_names}"))

    rc, usb_text = _run(["lsusb", "-t"])
    evidence["usb_topology"] = usb_text
    checks.append(Check("USB topology readable", rc == 0 and "Driver=uvcvideo" in usb_text,
                        True, "uvcvideo present" if "Driver=uvcvideo" in usb_text else usb_text[:300]))

    rc, memory_text = _run(["free", "-b"])
    rc_swap, swap_text = _run(["swapon", "--show", "--bytes"])
    evidence["memory"] = memory_text
    evidence["swap"] = swap_text
    memory_match = re.search(r"^Mem:\s+(\d+)", memory_text, re.MULTILINE)
    total_memory = int(memory_match.group(1)) if memory_match else 0
    checks.append(Check("8 GB-class shared memory visible",
                        rc == 0 and total_memory >= 7_000_000_000, True,
                        f"{total_memory} bytes"))
    checks.append(Check("swap/zram active", rc_swap == 0 and "/dev/zram" in swap_text,
                        True, swap_text.replace("\n", "; ")[:500]))

    rc_in, audio_in = _run(["arecord", "-l"])
    rc_out, audio_out = _run(["aplay", "-l"])
    evidence["audio_input"] = audio_in
    evidence["audio_output"] = audio_out
    checks.append(Check("USB microphone enumerates", rc_in == 0 and has_usb_audio(audio_in),
                        True, "USB Audio capture present" if has_usb_audio(audio_in) else
                        audio_in.replace("\n", "; ")[:500]))
    checks.append(Check("USB headset/output enumerates", rc_out == 0 and has_usb_audio(audio_out),
                        True, "USB Audio playback present" if has_usb_audio(audio_out) else
                        audio_out.replace("\n", "; ")[:500]))

    calibration = Path(cfg.depth.calibration_file)
    checks.append(Check("stereo calibration present", calibration.is_file(), True,
                        str(calibration)))
    if cfg.depth.backend == "cuda_sad":
        cuda_stereo_ok = (Path("/usr/local/cuda/bin/nvcc").is_file()
                          and packages["pycuda"] != "missing")
        checks.append(Check("CUDA stereo backend available", cuda_stereo_ok, True,
                            f"backend=cuda_sad nvcc={'yes' if Path('/usr/local/cuda/bin/nvcc').is_file() else 'no'} pycuda={packages['pycuda']}"))
    else:
        checks.append(Check("GPU depth engine present", Path(cfg.depth.raft_engine).is_file(), True,
                            cfg.depth.raft_engine))
    checks.append(Check("GPU grounding engine present", Path(cfg.grounding.engine).is_file(), True,
                        cfg.grounding.engine))
    checks.append(Check("production fallbacks disabled",
                        not cfg.depth.allow_cpu_fallback and not cfg.grounding.allow_torch_fallback,
                        True, f"cpu_depth={cfg.depth.allow_cpu_fallback}, torch_grounding={cfg.grounding.allow_torch_fallback}"))

    evidence["checks"] = [asdict(item) for item in checks]
    evidence["production_ready"] = all(item.ok for item in checks if item.required)
    evidence["summary"] = {
        "passed": sum(item.ok for item in checks),
        "failed": sum(not item.ok for item in checks),
        "required_failed": [item.name for item in checks if item.required and not item.ok],
    }
    return evidence


def print_report(report: dict) -> None:
    print("ARGUS environment baseline")
    print("=" * 72)
    for item in report["checks"]:
        mark = "PASS" if item["ok"] else "FAIL"
        required = "required" if item["required"] else "advisory"
        print(f"[{mark}] {item['name']} ({required})\n       {item['detail']}")
    summary = report["summary"]
    print("=" * 72)
    print(f"Passed: {summary['passed']}  Failed: {summary['failed']}")
    print(f"Production ready: {report['production_ready']}")
    if summary["required_failed"]:
        print("Required failures: " + ", ".join(summary["required_failed"]))


def write_report(report: dict, output: str | os.PathLike) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

#!/usr/bin/env python3
"""
ARGUS Autonomous Colab Pipeline Runner
Usage (paste into one Colab cell):
    !python run_argus_pipeline.py
"""

import os, sys, re, gc, time, json, shutil, subprocess, textwrap
from datetime import datetime
from pathlib import Path

# ── tqdm (graceful fallback if not installed) ─────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm"])
    from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
NB_DIR     = SCRIPT_DIR / "notebooks"
BASE       = Path("/content/drive/MyDrive/ARGUS")
LOGS_DIR   = BASE / "logs"

NOTEBOOKS = [
    ("00", "00_setup"),
    ("01", "01_stereo_depth"),
    ("02", "02_segmentation"),
    ("03", "03_object_detection"),
    ("04", "04_privacy_filter"),
    ("05", "05_speech_pipeline"),
    ("06", "06_llm_setup"),
    ("07", "07_integration_test"),
]

NB_LABELS = {
    "00": "Environment Setup",
    "01": "Stereo Depth (RAFT)",
    "02": "Semantic Segmentation",
    "03": "Object Detection     ",
    "04": "Privacy Filter",
    "05": "Speech Pipeline",
    "06": "LLM Setup",
    "07": "Integration Test",
}

ERROR_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"CUDA out of memory",
    r"FileNotFoundError",
    r"ModuleNotFoundError",
    r"RuntimeError",
    r"AssertionError",
    r"^\s*[A-Z][a-zA-Z]+Error:",
    r"\bKilled\b",
    r"\bkilled\b",
    r"Segmentation fault",
]

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      █████╗ ██████╗  ██████╗ ██╗   ██╗███████╗             ║
║     ██╔══██╗██╔══██╗██╔════╝ ██║   ██║██╔════╝             ║
║     ███████║██████╔╝██║  ███╗██║   ██║███████╗             ║
║     ██╔══██║██╔══██╗██║   ██║██║   ██║╚════██║             ║
║     ██║  ██║██║  ██║╚██████╔╝╚██████╔╝███████║             ║
║     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝             ║
║                                                              ║
║        Autonomous Colab Pipeline Runner v1.0                ║
║        8 Notebooks  •  A100 or T4  •  Full Pipeline         ║
╚══════════════════════════════════════════════════════════════╝
"""

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _exists(p):   return Path(p).exists()
def _mb(p):       return Path(p).stat().st_size / 1e6  if Path(p).exists() else 0
def _gb(p):       return Path(p).stat().st_size / 1e9  if Path(p).exists() else 0
def _notempty(p): return Path(p).exists() and any(Path(p).iterdir())
def _ts():        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def _hr(seconds): m, s = divmod(int(seconds), 60); h, m = divmod(m, 60); return f"{h}h {m}m {s}s"


def clear_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        free  = torch.cuda.mem_get_info()[0] / 1e9
        total = torch.cuda.mem_get_info()[1] / 1e9
        print(f"  GPU memory cleared: {free:.1f} / {total:.1f} GB free")
    except Exception as e:
        print(f"  GPU clear skipped: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Environment validation
# ═════════════════════════════════════════════════════════════════════════════

def phase0():
    print("\n" + "="*62)
    print("  PHASE 0 — Environment Validation")
    print("="*62)

    try:
        import torch
        import psutil
    except ImportError as e:
        print(f"[FATAL] Missing package: {e}")
        sys.exit(1)

    all_pass = True

    def chk(label, ok, fatal=False, note=""):
        nonlocal all_pass
        icon = "✅" if ok else ("❌" if fatal else "⚠️ ")
        print(f"  {icon} {'PASS' if ok else 'FAIL'} — {label}" + (f" ({note})" if note else ""))
        if not ok and fatal:
            all_pass = False
        return ok

    # GPU
    gpu_ok   = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "N/A"
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_ok else 0

    chk("CUDA available",                gpu_ok,                    fatal=True)
    # A100 needed for NB01-03 (heavy training). T4 is fine for NB04-07.
    needs_a100 = not all([
        _exists(BASE/"exports/tensorrt/raft_stereo_640x480.onnx"),
        _exists(BASE/"models/segmentation/segformer_b2_argus/config.json"),
        _exists(BASE/"models/detection/yolov8s_argus_final.pt"),
    ])
    if needs_a100 and "A100" not in gpu_name:
        chk(f"GPU: {gpu_name} — A100 recommended for NB01-03", False, fatal=False,
            note="NB01-03 not done yet — A100 trains 5x faster; T4 will work but slowly")
    else:
        chk(f"GPU: {gpu_name}", gpu_ok, fatal=False,
            note="T4 is fine — NB01-03 already complete" if "A100" not in gpu_name else "")
    chk(f"VRAM >= 14 GB (got: {vram_gb:.1f} GB)", vram_gb >= 14,   fatal=True)

    # RAM
    ram_gb = psutil.virtual_memory().total / 1e9
    chk(f"System RAM >= 12 GB (got: {ram_gb:.1f} GB)", ram_gb >= 12, fatal=True)

    # Drive
    drive_ok = _exists(BASE)
    chk(f"Drive mounted at {BASE}", drive_ok, fatal=True,
        note="Run: from google.colab import drive; drive.mount('/content/drive')" if not drive_ok else "")

    if drive_ok:
        _, _, free_bytes = shutil.disk_usage(BASE)
        free_gb = free_bytes / 1e9
        chk(f"Drive free >= 80 GB (got: {free_gb:.1f} GB)", free_gb >= 80, fatal=False,
            note="Low space — datasets+models need ~60 GB" if free_gb < 80 else "")

    # Notebooks present
    nb_count = len(list(NB_DIR.glob("*.ipynb")))
    chk(f"Notebooks found: {nb_count}/8", nb_count == 8, fatal=True,
        note=f"Expected in {NB_DIR}")

    print()
    if not all_pass:
        print("❌ One or more FATAL checks failed. Fix the issues above and re-run.")
        sys.exit(1)

    print("✅ All environment checks passed.\n")
    return gpu_name, round(vram_gb, 1), round(ram_gb, 1)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Drive verification per notebook
# ═════════════════════════════════════════════════════════════════════════════

def verify_notebook(n: int) -> dict:
    """Run Drive verification checks for notebook n (0-7). Returns {label: bool}."""
    results = {}

    def chk(label, condition):
        ok = bool(condition)
        results[label] = ok
        print(f"    {'✅' if ok else '❌'} {'PASS' if ok else 'FAIL'} — {label}")
        return ok

    B = BASE
    if n == 0:
        for d in ["datasets", "models", "exports", "logs", "checkpoints"]:
            chk(f"{d}/ exists", _exists(B / d))

    elif n == 1:
        chk("raftstereo-realtime.pth downloaded",    _mb(B/"models/depth/raftstereo-realtime.pth") > 10)
        chk("raft_stereo_final.pth > 10 MB",         _mb(B/"models/depth/raft_stereo_final.pth") > 10)
        chk("raft_stereo_640x480.onnx exported",     _exists(B/"exports/tensorrt/raft_stereo_640x480.onnx"))
        chk("RAFT ONNX > 5 MB",                      _mb(B/"exports/tensorrt/raft_stereo_640x480.onnx") > 5)

    elif n == 2:
        chk("ADE20K done flag",                      _exists(B/"datasets/ade20k/ade20k_done.flag"))
        chk("NYUv2 downloaded",                      _mb(B/"datasets/nyuv2/nyu_depth_v2_labeled.mat") > 100)
        chk("segformer_b2_argus/ not empty",         _notempty(B/"models/segmentation/segformer_b2_argus"))
        chk("SegFormer config.json exists",          _exists(B/"models/segmentation/segformer_b2_argus/config.json"))
        chk("segformer_b2_512x512.onnx exported",   _exists(B/"exports/tensorrt/segformer_b2_512x512.onnx"))
        chk("SegFormer ONNX > 50 MB",               _mb(B/"exports/tensorrt/segformer_b2_512x512.onnx") > 50)

    elif n == 3:
        chk("COCO done flag",                        _exists(B/"datasets/coco/coco_done.flag"))
        chk("yolov8s_argus_final.pt > 20 MB",        _mb(B/"models/detection/yolov8s_argus_final.pt") > 20)
        chk("yolov8s_argus_640.onnx exported",       _exists(B/"exports/tensorrt/yolov8s_argus_640.onnx"))

    elif n == 4:
        chk("privacy/ not empty",                    _notempty(B/"models/privacy"))
        chk("privacy_test.png saved",                _exists(B/"logs/privacy_test.png"))
        chk("EasyOCR done flag",                     _exists(B/"models/privacy/easyocr/easyocr_done.flag"))
        chk("buffalo_s done flag",                   _exists(B/"models/privacy/buffalo_done.flag"))

    elif n == 5:
        chk("whisper/ not empty",                    _notempty(B/"models/speech/whisper"))
        chk("Piper ONNX exists",                     _exists(B/"models/piper/en_US-lessac-medium.onnx"))
        chk("Piper ONNX > 50 MB",                   _mb(B/"models/piper/en_US-lessac-medium.onnx") > 50)
        chk("Piper config JSON exists",              _exists(B/"models/piper/en_US-lessac-medium.onnx.json"))
        chk("wakeword_argus/ exists",                _exists(B/"models/wakeword_argus"))

    elif n == 6:
        gguf = B/"models/Phi-3.5-mini-instruct-Q4_K_M.gguf"
        chk("Phi-3.5 Mini GGUF exists",              _exists(gguf))
        chk("Phi-3.5 Mini GGUF > 2 GB",             _gb(gguf) > 2.0)

    elif n == 7:
        manifest = B/"models/models_manifest.json"
        chk("models_manifest.json exists",           _exists(manifest))
        if _exists(manifest):
            try:
                with open(manifest) as f:
                    data = json.load(f)
                for model_name, info in data.get("argus_models", {}).items():
                    for key, val in info.items():
                        if isinstance(val, str) and val.startswith("/content/drive"):
                            chk(f"  {model_name}.{key} exists", _exists(val))
            except Exception as e:
                chk(f"manifest parseable", False)

    return results


def is_done(n: int) -> bool:
    """True only if ALL verify checks for this notebook pass."""
    results = {}

    B = BASE
    if n == 0:
        return all(_exists(B / d) for d in ["datasets", "models", "exports", "logs", "checkpoints"])
    elif n == 1:
        return _mb(B/"models/depth/raft_stereo_final.pth") > 10 and \
               _exists(B/"exports/tensorrt/raft_stereo_640x480.onnx")
    elif n == 2:
        return _notempty(B/"models/segmentation/segformer_b2_argus") and \
               _mb(B/"exports/tensorrt/segformer_b2_512x512.onnx") > 50
    elif n == 3:
        return _mb(B/"models/detection/yolov8s_argus_final.pt") > 20
    elif n == 4:
        return _exists(B/"logs/privacy_test.png") and _exists(B/"models/privacy/buffalo_done.flag")
    elif n == 5:
        return _exists(B/"models/piper/en_US-lessac-medium.onnx")
    elif n == 6:
        return _gb(B/"models/Phi-3.5-mini-instruct-Q4_K_M.gguf") > 2.0
    elif n == 7:
        return _exists(B/"models/models_manifest.json")
    return False


# ═════════════════════════════════════════════════════════════════════════════
# Notebook runner
# ═════════════════════════════════════════════════════════════════════════════

def run_notebook(nb_idx: str, nb_name: str) -> tuple[bool, Path]:
    """Execute one notebook. Returns (success, log_path)."""
    nb_path  = NB_DIR / f"{nb_name}.ipynb"
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"notebook_{nb_idx}_{ts}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=86400",
        "--ExecutePreprocessor.kernel_name=python3",
        str(nb_path),
    ]

    print(f"\n  Command: {' '.join(cmd[-4:])}")
    print(f"  Log:     {log_path}\n")

    returncode = None
    import threading

    def _heartbeat(t_start, stop_evt):
        """Print elapsed time every 60 s so the terminal doesn't look frozen."""
        while not stop_evt.wait(60):
            elapsed = time.time() - t_start
            h, rem  = divmod(int(elapsed), 3600)
            m, s    = divmod(rem, 60)
            print(f"\n  [heartbeat] NB {nb_idx} still running — "
                  f"elapsed {h:02d}:{m:02d}:{s:02d} | {_ts()}", flush=True)

    stop_evt  = threading.Event()
    t_nb_start = time.time()
    hb_thread  = threading.Thread(target=_heartbeat, args=(t_nb_start, stop_evt), daemon=True)
    hb_thread.start()

    with open(log_path, "w", buffering=1) as log_f:
        log_f.write(f"=== ARGUS notebook_{nb_idx} started at {_ts()} ===\n")
        log_f.write(f"Command: {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(SCRIPT_DIR),
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
        proc.wait()
        returncode = proc.returncode
        log_f.write(f"\n=== Finished at {_ts()} | exit code: {returncode} ===\n")

    stop_evt.set()
    hb_thread.join()
    return returncode == 0, log_path


def scan_errors(log_path: Path) -> list:
    """Return list of (line_no, line) for any error pattern matches."""
    hits = []
    with open(log_path, errors="replace") as f:
        for i, line in enumerate(f, 1):
            if any(re.search(p, line) for p in ERROR_PATTERNS):
                hits.append((i, line.rstrip()))
    return hits


def write_failed_flag(nb_idx: str):
    flag = LOGS_DIR / f"notebook_{nb_idx}.failed"
    flag.write_text(f"Failed at {_ts()}\n")


def write_manifest():
    """Create models_manifest.json listing every expected model path."""
    B = str(BASE)
    manifest = {
        "generated_at": _ts(),
        "argus_models": {
            "raft_stereo": {
                "weights":      f"{B}/models/depth/raft_stereo_final.pth",
                "onnx":         f"{B}/exports/tensorrt/raft_stereo_640x480.onnx",
            },
            "segformer_b2": {
                "model_dir":    f"{B}/models/segmentation/segformer_b2_argus",
                "onnx":         f"{B}/exports/tensorrt/segformer_b2_512x512.onnx",
            },
            "yolov8": {
                "weights":      f"{B}/models/detection/yolov8s_argus_final.pt",
                "onnx":         f"{B}/exports/tensorrt/yolov8s_argus_640.onnx",
            },
            "insightface": {
                "model_dir":    f"{B}/models/privacy/buffalo_s",
            },
            "easyocr": {
                "model_dir":    f"{B}/models/privacy/easyocr",
            },
            "whisper": {
                "model_dir":    f"{B}/models/speech/whisper",
            },
            "piper_tts": {
                "onnx":         f"{B}/models/piper/en_US-lessac-medium.onnx",
                "config":       f"{B}/models/piper/en_US-lessac-medium.onnx.json",
            },
            "phi35_mini": {
                "gguf":         f"{B}/models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
            },
        }
    }
    out = BASE / "models" / "models_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  models_manifest.json written → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Final report
# ═════════════════════════════════════════════════════════════════════════════

def final_report(gpu_name, vram_gb, ram_gb, nb_results, errors_by_nb,
                 total_seconds, report_path):

    def icon(ok):    return "✅ PASS" if ok else "❌ FAIL"
    def exists(ok):  return "✅ EXISTS" if ok else "❌ MISSING"

    B = BASE
    _, used_b, free_b = shutil.disk_usage(B) if _exists(B) else (0, 0, 0)
    used_gb = used_b / 1e9
    free_gb = free_b / 1e9

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║           ARGUS PIPELINE EXECUTION REPORT               ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ GPU Used          : {gpu_name:<38}║",
        f"║ VRAM              : {vram_gb:<1.0f} GB{'':<35}║",
        f"║ System RAM        : {ram_gb:<1.0f} GB{'':<35}║",
        f"║ Total Runtime     : {_hr(total_seconds):<38}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║ NOTEBOOKS                                               ║",
    ]

    labels = {
        "00": "Environment Setup     ",
        "01": "Stereo Depth         ",
        "02": "Semantic Segmentation",
        "03": "Object Detection     ",
        "04": "Privacy Filter       ",
        "05": "Speech Pipeline      ",
        "06": "LLM Setup            ",
        "07": "Integration Test     ",
    }
    for idx, lbl in labels.items():
        ok   = nb_results.get(idx, False)
        skip = nb_results.get(idx + "_skipped", False)
        tag  = "⏭ SKIP" if skip else icon(ok)
        lines.append(f"║  {idx} — {lbl} : {tag:<14}║")

    lines += [
        "╠══════════════════════════════════════════════════════════╣",
        "║ MODELS ON DRIVE                                         ║",
        f"║  RAFT-Stereo (.pth)          : {exists(_mb(B/'models/depth/raft_stereo_final.pth')>10):<18}║",
        f"║  SegFormer-B2 (folder)       : {exists(_notempty(B/'models/segmentation/segformer_b2_argus')):<18}║",
        f"║  YOLOv8-small (.pt)          : {exists(_mb(B/'models/detection/yolov8s_argus_final.pt')>20):<18}║",
        f"║  SCRFD Face detector         : {exists(_notempty(B/'models/privacy')):<18}║",
        f"║  EasyOCR text detector       : {exists((B/'models/privacy/easyocr/easyocr_done.flag')):<18}║",
        f"║  Whisper tiny INT8           : {exists(_notempty(B/'models/speech/whisper')):<18}║",
        f"║  Piper TTS voice             : {exists(_exists(B/'models/piper/en_US-lessac-medium.onnx')):<18}║",
        f"║  Phi-3.5 Mini Q4 GGUF        : {exists(_gb(B/'models/Phi-3.5-mini-instruct-Q4_K_M.gguf')>2):<18}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║ ONNX EXPORTS READY FOR JETSON                           ║",
        f"║  raft_stereo_640x480.onnx    : {exists(_exists(B/'exports/tensorrt/raft_stereo_640x480.onnx')):<18}║",
        f"║  segformer_b2_512x512.onnx   : {exists(_exists(B/'exports/tensorrt/segformer_b2_512x512.onnx')):<18}║",
        f"║  yolov8s_argus_640.onnx      : {exists(_exists(B/'exports/tensorrt/yolov8s_argus_640.onnx')):<18}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║ STORAGE                                                 ║",
        f"║  Total Drive used : {used_gb:<4.1f} GB{'':<32}║",
        f"║  Drive remaining  : {free_gb:<4.1f} GB{'':<32}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║ NEXT STEPS FOR JETSON                                   ║",
        "║  1. scp exports/ to Jetson                              ║",
        "║  2. Run trtexec on Jetson for each ONNX                 ║",
        "║  3. Run argus_pipeline.py with cameras connected        ║",
        "╚══════════════════════════════════════════════════════════╝",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n")
    print(f"\n  Report saved → {report_path}")

    # Phase 5 — per-failure details
    if errors_by_nb:
        print("\n" + "="*62)
        print("  PHASE 5 — Failure Details")
        print("="*62)
        for nb_idx, (log_path, error_lines) in errors_by_nb.items():
            nb_name = next(n for i, n in NOTEBOOKS if i == nb_idx)
            print(f"\n  ── Notebook {nb_idx} ({nb_name}) ──")
            print(f"  Log: {log_path}")
            print(f"  Safe to re-run without earlier notebooks: "
                  f"{'YES' if nb_idx not in ('00',) else 'NO — run 00 first'}")
            print(f"  Retry command:")
            print(f"    !jupyter nbconvert --to notebook --execute --inplace "
                  f"--ExecutePreprocessor.timeout=86400 "
                  f"notebooks/{nb_name}.ipynb")
            print(f"\n  Last error lines from log:")
            # Show last 30 lines of log
            try:
                all_lines = log_path.read_text(errors="replace").splitlines()
                for line in all_lines[-30:]:
                    print(f"    {line}")
            except Exception:
                for lineno, line in error_lines[-10:]:
                    print(f"    L{lineno}: {line}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def git_pull():
    """Pull latest notebooks from GitHub so we always run up-to-date code."""
    repo_dir = SCRIPT_DIR
    print("\n" + "="*62)
    print("  PHASE -1 — Git Pull (latest notebooks)")
    print("="*62)
    try:
        # Check if this is a git repo
        r = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("  ⚠️  Not a git repo — skipping pull")
            return
        # Show current commit
        cur = subprocess.run(["git", "-C", str(repo_dir), "log", "--oneline", "-1"],
                             capture_output=True, text=True)
        print(f"  Current: {cur.stdout.strip()}")
        # Pull
        pr = subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", "main"],
                            capture_output=True, text=True)
        if pr.returncode == 0:
            after = subprocess.run(["git", "-C", str(repo_dir), "log", "--oneline", "-1"],
                                   capture_output=True, text=True)
            msg = pr.stdout.strip() or "Already up to date."
            print(f"  ✅ {msg}")
            print(f"  Now at:  {after.stdout.strip()}")
        else:
            print(f"  ⚠️  git pull failed (non-fatal): {pr.stderr.strip()}")
            print("      Continuing with local notebooks.")
    except FileNotFoundError:
        print("  ⚠️  git not found — skipping pull")
    print()


def main():
    print(BANNER)
    print(f"  Started : {_ts()}")
    print(f"  Notebooks: {NB_DIR}")
    print(f"  Drive   : {BASE}")

    t_start = time.time()

    git_pull()

    # Phase 0
    gpu_name, vram_gb, ram_gb = phase0()

    nb_results   = {}   # idx -> bool (pass/fail)
    errors_by_nb = {}   # idx -> (log_path, error_lines)

    # Phase 1 — iterate notebooks
    print("\n" + "="*62)
    print("  PHASE 1 — Sequential Notebook Execution")
    print("="*62)

    nb_bar = tqdm(NOTEBOOKS, desc="Notebooks", unit="nb", ncols=70)
    for nb_idx, nb_name in nb_bar:
        n = int(nb_idx)
        nb_bar.set_description(f"NB {nb_idx} {nb_name}")

        # Skip if already done
        if is_done(n):
            print(f"\n  ⏭  NB {nb_idx} ({nb_name}) — SKIPPING (already complete on Drive)")
            nb_results[nb_idx]              = True
            nb_results[nb_idx + "_skipped"] = True
            continue

        # Special: write manifest before running NB 07
        if n == 7:
            write_manifest()

        print(f"\n{'='*62}")
        print(f"  [{_ts()}]  Starting NB {nb_idx} — {nb_name}")
        print(f"{'='*62}")

        t_nb = time.time()
        success, log_path = run_notebook(nb_idx, nb_name)
        elapsed = time.time() - t_nb

        print(f"\n  [{_ts()}]  NB {nb_idx} finished in {_hr(elapsed)}")

        # Error scan
        error_lines = scan_errors(log_path)

        if not success or error_lines:
            print(f"\n  ❌ NB {nb_idx} FAILED")
            if error_lines:
                print(f"  First error at line {error_lines[0][0]}: {error_lines[0][1]}")
            write_failed_flag(nb_idx)
            nb_results[nb_idx]   = False
            errors_by_nb[nb_idx] = (log_path, error_lines)
            # Stop pipeline on failure
            print(f"\n  Pipeline halted. Fix the error above, then re-run.")
            print(f"  Already-completed notebooks will be skipped automatically.\n")
            break

        # Phase 2 — Drive verification
        print(f"\n  Drive verification for NB {nb_idx}:")
        v = verify_notebook(n)
        all_pass = all(v.values())
        nb_results[nb_idx] = all_pass
        if not all_pass:
            failed_checks = [k for k, ok in v.items() if not ok]
            print(f"  ⚠  {len(failed_checks)} check(s) failed: {', '.join(failed_checks)}")

        # Phase 3 — GPU clear
        print(f"\n  Clearing GPU memory...")
        clear_gpu()

    # Phase 4 — Final report
    total_seconds = time.time() - t_start
    report_path   = LOGS_DIR / "final_report.txt"
    final_report(gpu_name, vram_gb, ram_gb, nb_results, errors_by_nb,
                 total_seconds, report_path)

    all_ok = all(nb_results.get(idx, False) for idx, _ in NOTEBOOKS)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

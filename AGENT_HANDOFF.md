# ARGUS — Agent Handoff Document

> **Purpose:** This document captures the full development history of the ARGUS ML pipeline — what was built, what broke, why it broke, and how it was fixed. Use this to onboard a new Claude session without repeating mistakes.

---

## 1. Project Overview

**ARGUS** is an AI-powered smart glasses system for the visually impaired.

- **Training environment:** Google Colab Pro+ (A100 / L4 / T4 GPU — see GPU tier table below)
- **Local machine:** GTX 1060 6GB, 150 GB free on D:\ (Windows) — use for debugging, data prep, calibration
- **Deployment target:** NVIDIA Jetson Orin Nano Super (8 GB)
- **Cameras:**
  - 2× Arducam AR0234 2.3MP Global Shutter USB 3.0 (stereo depth)
  - 1× Arducam IMX477P 12MP USB 3.0 M12 lens (wide scene)
- **GitHub repo:** `https://github.com/meteorboyF/Argus.git`
- **Drive storage:** `/content/drive/MyDrive/ARGUS/` (models, exports, datasets)
- **Repo clone path on Colab:** `/content/Argus/` (notebooks, pipeline script)

### Pipeline entry point
```bash
%run /content/Argus/run_argus_pipeline.py
```
The script auto-pulls from GitHub, checks hardware, then runs NB01–NB07 in order, skipping any already completed.

---

## 2. Hardware Context

### Colab Pro+ GPU Tier Reference

| GPU | CU/hr | VRAM | Architecture | Use For ARGUS | Avoid When |
|---|---|---|---|---|---|
| Standard CPU | ~0.00 | — | — | Syntax debug, pip installs, unzipping | Any model forward/backward pass |
| T4 | ~1.2–2.0 | 15 GB | Turing | NB04–07, lightweight inference tests, first-pass exploration | Large vision transformers (OOM risk) |
| L4 | ~1.7–3.5 | 22.5 GB | Ada Lovelace | Mid-tier fine-tuning, FP16/BF16 mixed precision, stable alternatives to A100 | Massive pre-training, batches >24GB VRAM |
| V100 (Legacy) | ~5.0–6.0 | 16 GB | Volta | Only if L4 unavailable | Modern FP8/BF16 pipelines — L4 beats it for fewer units |
| A100 | ~13.0–15.0 | 40–80 GB | Ampere | NB01–03 heavy training only | Debugging, small datasets, CPU-bottlenecked scripts |
| G4 / RTX 6000 | ~8.5–9.0 | 96 GB | Blackwell | Memory-bound tasks needing extreme VRAM | Tiny tasks — 96GB frame buffer is wasted |

> ⚠️ **A100 costs 7–10× more per hour than T4.** Never use it for anything except confirmed-working training runs.

### Decision Guide for ARGUS

```
Writing / debugging code?          → Local GTX 1060 or Colab CPU
First test of new notebook?        → T4
NB04–07 (privacy, speech, LLM)?    → T4 (15GB is sufficient)
NB01–03 training (confirmed code)? → A100 (40GB needed for batch size)
Need more than 15GB but <40GB?     → L4 (cheaper than A100, Ada architecture)
Memory-bound, model won't fit A100?→ G4 RTX 6000 (96GB)
```

### Local Machine (Windows)
- **GPU:** GTX 1060 6GB (CUDA 11.8 compatible)
- **Free storage:** 150 GB on D:\
- **Best uses:** Stereo calibration, data preprocessing, ONNX validation, small inference tests, debugging notebook logic before burning Colab units
- **PyTorch install:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Deployment
| Device | RAM | Role |
|---|---|---|
| Jetson Orin Nano Super | 8 GB unified | Inference only — no training |

**AR0234 cameras** use global shutter — no rolling shutter artifacts, critical for accurate stereo depth matching at walking speed.

---

## 3. Notebook Summary

| Notebook | Purpose | Status |
|---|---|---|
| NB01 `01_stereo_depth.ipynb` | Train RAFT-Stereo, export ONNX | ✅ Done |
| NB02 `02_segmentation.ipynb` | Train SegFormer-B2, export ONNX | ✅ Done |
| NB03 `03_object_detection.ipynb` | Fine-tune YOLOv8s on COCO, export | ✅ Done (burned 8 hrs — see mistakes) |
| NB04 `04_privacy_filter.ipynb` | InsightFace face blur + EasyOCR text blur | ⚠️ Fixes committed, needs re-run |
| NB05 `05_speech_pipeline.ipynb` | Whisper STT + Piper TTS + openWakeWord | ⚠️ Fixes committed, needs re-run |
| NB06 `06_llm_setup.ipynb` | Download Phi-3.5 GGUF for Jetson inference | ⚠️ Fixes committed, needs re-run |
| NB07 `07_integration_test.ipynb` | End-to-end pipeline smoke test | ❌ Not yet run |

---

## 4. Mistakes Made & How They Were Fixed

### 4.1 NB01 — ONNX Export: `dynamo=True` crash

**Error:**
```
TorchExportError: aten.cudnn_grid_sampler.default is not supported by torch.onnx.export
```

**Root cause:** PyTorch 2.10's dynamo-based ONNX exporter doesn't support `cudnn_grid_sampler`. RAFT-Stereo uses this op internally.

**Fix:** Set `dynamo=False`, `opset_version=16`, wrapped model in a thin `nn.Module`:
```python
class RAFTStereoExport(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, l, r): return self.model(l, r, iters=12, test_mode=True)

torch.onnx.export(
    RAFTStereoExport(model), (dummy_left, dummy_right), ONNX_PATH,
    dynamo=False, opset_version=16, input_names=["left","right"],
    output_names=["disparity"],
    dynamic_axes={"left":{0:"batch"},"right":{0:"batch"},"disparity":{0:"batch"}},
    do_constant_folding=True,
)
```

---

### 4.2 NB02 — albumentations v2 API mismatch

**Error:**
```
ValidationError: RandomResizedCrop — unexpected keyword / wrong positional args
```

**Root cause:** albumentations v2 changed `RandomResizedCrop(h, w, scale)` to require named `size=` and `scale` must be ≤ 1.0. We were passing `scale=(0.5, 2.0)` which is invalid (max 1.0), and `Resize` also changed to keyword-only.

**Fix:**
```python
A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0))
A.Resize(height=IMG_SIZE, width=IMG_SIZE)
```

---

### 4.3 NB03 — COCO training from Drive: 8-hour disaster

**What happened:** Training ran for 10+ hours instead of ~30 minutes. Burned ~45 compute units (≈ half a monthly Pro budget).

**Root cause:** The COCO YAML pointed to Drive (`/content/drive/MyDrive/Argus/datasets/coco/`). Google Drive FUSE reads each file in ~5ms. With 118K images × 50 epochs × 5ms per file read = ~8 hours of pure I/O overhead.

**Fix:** Extract ZIP files to local SSD (`/content/coco/`) before training, point YAML to local path:
```python
LOCAL_COCO = '/content/coco'
# Extract train2017.zip, val2017.zip, annotations.zip to LOCAL_COCO
# Then set COCO_YAML to point to LOCAL_COCO
```

**Secondary issue — Drive mount timeout:** After training, the Drive mount timed out because we tried to copy 118K extracted COCO images back to Drive. Drive FUSE has a ~10K files/folder limit before it hangs. 

**Fix:** Delete the extracted image folders from Drive. Keep only the ZIP files in Drive. Always extract fresh to `/content/` at the start of each session.

---

### 4.4 NB04 — Wikipedia URL 400 error + `cv2.cvtColor(None)` crash

**Error:**
```
urllib.error.HTTPError: HTTP Error 400: Bad Request
cv2.error: (-5) in function cvtColor — src is None
```

**Root cause:** The notebook fetched a test image from Wikipedia. Wikipedia returned 400 (bot detection / URL changed). `cv2.imread(None)` silently returns `None`, then `cvtColor` crashes.

**Fix:** Replaced external URL fetch with a synthetic PIL image generated in-memory:
```python
pil_img = Image.new('RGB', (640, 480), color=(180, 160, 140))
draw = ImageDraw.Draw(pil_img)
draw.ellipse([220, 100, 420, 340], fill=(220, 185, 150))  # synthetic face
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

---

### 4.5 NB04 — EasyOCR bbox TypeError

**Error:**
```
TypeError: cannot unpack non-iterable numpy.int64 object
```

**Root cause:** Newer EasyOCR versions return bounding boxes as nested arrays (list of 4 corner points), not flat `[x_min, y_min, x_max, y_max]`.

**Fix:**
```python
for bbox in boxes:
    coords = np.array(bbox).flatten().astype(int)
    x_min, x_max, y_min, y_max = coords[0], coords[1], coords[2], coords[3]
```

---

### 4.6 NB05 — pyaudio build failure

**Error:** `portaudio.h: No such file or directory`

**Root cause:** pyaudio requires system `portaudio` dev headers which aren't available in the standard Colab image.

**Fix:** Removed pyaudio entirely from the notebook. Colab is headless — there's no microphone anyway. Audio I/O only matters on Jetson.

---

### 4.7 NB05 — Piper TTS path mismatch

**Error:** Pipeline couldn't find Piper TTS after NB05 completed.

**Root cause:** NB05 saved Piper to `models/speech/piper/` but `run_argus_pipeline.py` checked for `models/piper/`.

**Fix:** Changed NB05:
```python
TTS_DIR = f'{BASE}/models/piper'        # was: f'{MODELS}/piper'
WAKE_OUT = f'{BASE}/models/wakeword_argus'  # was: f'{MODELS}/wakeword_argus'
```

---

### 4.8 NB06 — llama-cpp-python build failure (Python 3.12)

**Error:**
```
CMake Error: target_compile_options called with invalid arguments
No pre-built wheel available for Python 3.12
```

**Root cause:** llama-cpp-python's old `-DLLAMA_CUBLAS=on` CMake flag is deprecated. Also, Colab now runs Python 3.12 which has no pre-built wheel. Building from source takes 20+ minutes and still fails.

**Fix:** Skip llama-cpp-python entirely on Colab. Just download the GGUF model file. Actual inference runs on Jetson (ARM build works fine there):
```python
# Skip: from llama_cpp import Llama  — inference is on Jetson
# Just download the GGUF:
!huggingface-cli download microsoft/Phi-3.5-mini-instruct-gguf \
    Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir {LLM_DIR}
```

---

### 4.9 `run_argus_pipeline.py` — T4 GPU blocked by phase0()

**Error:** Pipeline refused to run on T4, even though T4 is fine for NB04–07.

**Root cause:** phase0() had hardcoded checks: `A100 required`, `VRAM >= 38 GB`, `RAM >= 50 GB`.

**Fix:** Smart GPU check — only warn (not fail) if T4 is used but NB01–03 are already done. Thresholds changed to 14 GB VRAM / 12 GB RAM:
```python
needs_a100 = not all([
    _exists(BASE/"exports/tensorrt/raft_stereo_640x480.onnx"),
    _exists(BASE/"models/segmentation/segformer_b2_argus/config.json"),
    _exists(BASE/"models/detection/yolov8s_argus_final.pt"),
])
if needs_a100 and "A100" not in gpu_name:
    chk(f"GPU: {gpu_name} — A100 recommended for NB01-03", False, fatal=False)
else:
    chk(f"GPU: {gpu_name}", gpu_ok, fatal=False)
chk(f"VRAM >= 14 GB", vram_gb >= 14, fatal=True)
chk(f"RAM >= 12 GB", ram_gb >= 12, fatal=True)
```

---

### 4.10 Git pull silently failing (nbconvert `--inplace`)

**Problem:** Committed fixes to notebooks weren't being applied on Colab. Old broken code kept running.

**Root cause:** Jupyter's `nbconvert --inplace` writes cell outputs back into the `.ipynb` files, creating local modifications. `git pull --ff-only` refuses to overwrite local changes, fails silently, and the old broken notebooks keep running.

**First attempted fix:** `git checkout -- notebooks/` before pull. This worked for the notebooks but not for `run_argus_pipeline.py` itself (if the script has its own local changes, it can't pull its own fix).

**Final fix:** Replaced `git pull --ff-only` with `git fetch + git reset --hard origin/main`:
```python
subprocess.run(["git", "-C", repo_dir, "fetch", "origin", "main"], ...)
subprocess.run(["git", "-C", repo_dir, "reset", "--hard", "origin/main"], ...)
```
This guarantees the entire working tree matches remote, regardless of any local dirty state.

---

### 4.11 ARGUS Drive folder is not a git repo

**Discovery:** `/content/drive/MyDrive/ARGUS/` is just a plain folder — it was never `git clone`d. It contains only large binary files (models, exports, datasets, calibration data). All code (notebooks, pipeline script) must be cloned fresh from GitHub each session.

**Correct workflow:**
```python
# Cell 1 — mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — clone code (notebooks live here, not on Drive)
!git clone https://github.com/meteorboyF/Argus.git /content/Argus

# Cell 3 — run pipeline
%run /content/Argus/run_argus_pipeline.py
```

---

## 5. Architecture of `run_argus_pipeline.py`

Key constants:
```python
BASE     = Path("/content/drive/MyDrive/ARGUS")   # Drive — persists across sessions
NB_DIR   = SCRIPT_DIR / "notebooks"               # local clone — lost on disconnect
SCRIPT_DIR = Path(__file__).parent                 # /content/Argus/
```

Key functions:
- `git_pull()` — fetch + reset --hard to remote
- `phase0()` — hardware check (GPU, VRAM, RAM, Drive mount, disk space)
- `run_nb(n)` — executes notebook via nbconvert, saves outputs inplace
- `is_done(n)` — checks Drive for completion artifacts (flags/model files)
- `verify_notebook(n)` — post-run validation of outputs

Completion flags on Drive:
| NB | Flag/artifact checked |
|---|---|
| NB01 | `exports/tensorrt/raft_stereo_640x480.onnx` |
| NB02 | `models/segmentation/segformer_b2_argus/config.json` |
| NB03 | `models/detection/yolov8s_argus_final.pt` |
| NB04 | `models/privacy/buffalo_done.flag` |
| NB05 | `models/piper/` + `models/wakeword_argus/` |
| NB06 | `models/llm/` (GGUF file) |
| NB07 | `logs/integration_test_done.flag` |

---

## 6. What's Left To Do

### Immediate (Colab)
- [ ] Re-run NB04 (privacy filter) — fixes committed
- [ ] Re-run NB05 (speech pipeline) — Piper path fix committed
- [ ] Re-run NB06 (LLM setup) — llama-cpp skip + GGUF download committed
- [ ] Run NB07 (integration test) — first time

### After pipeline completes
- [ ] Write stereo calibration script for AR0234 cameras (checkerboard pattern, save intrinsics + extrinsics to Drive)
- [ ] Verify all model files are on Drive before Jetson transfer

### Jetson Orin Nano Super setup
- [ ] Install JetPack 6
- [ ] Install ARM-compatible PyTorch (from NVIDIA Jetson wheels, not pip)
- [ ] Install ultralytics, onnxruntime-gpu, insightface, easyocr, piper-tts, openWakeWord
- [ ] Build llama-cpp-python with CUDA on ARM — this works on Jetson unlike Colab
- [ ] Transfer models from Drive to Jetson via `scp`
- [ ] Convert ONNX models to TensorRT engines on Jetson (must be done on-device)
- [ ] Write real-time pipeline orchestration layer

### Jetson TensorRT conversion (run on-device)
```bash
trtexec --onnx=raft_stereo_640x480.onnx --saveEngine=raft_stereo.trt --fp16
trtexec --onnx=segformer_b2_argus.onnx  --saveEngine=segformer.trt  --fp16
```

---

## 7. Model Choices & Rationale

Models were selected for the RTX 5060 Ti 16GB originally but A100 was used for training (better, not worse).

| Component | Model | Why |
|---|---|---|
| Stereo depth | RAFT-Stereo | Best accuracy on ETH3D benchmark; global shutter cameras eliminate rolling shutter errors |
| Segmentation | SegFormer-B2 | Good accuracy/speed tradeoff; B2 fits in Jetson RAM with TensorRT |
| Object detection | YOLOv8s | Real-time on Jetson; s-size fits 8GB; fine-tuned on COCO 118K images |
| Face/text blur | InsightFace + EasyOCR | Privacy requirement; runs at inference only |
| Wake word | openWakeWord | Lightweight, runs on CPU |
| STT | Whisper (small/base) | Good accuracy in noise; fits Jetson RAM |
| TTS | Piper | Fastest neural TTS; low latency on ARM |
| LLM | Phi-3.5-mini Q4_K_M GGUF | 2.3GB; fits Jetson 8GB unified memory alongside other models |

---

## 8. Compute Budget Tracking

**Plan:** Colab Pro+ (~500 CU/month, priority A100 access)

| Event | GPU | CU/hr | Hours | CU used | Notes |
|---|---|---|---|---|---|
| NB01 RAFT-Stereo training | A100 | ~14 | ~1 | ~14 CU | Normal |
| NB02 SegFormer training | A100 | ~14 | ~2 | ~28 CU | Normal |
| NB03 — COCO from Drive | A100 | ~14 | ~10 | ~140 CU | ❌ **WASTED** — Drive FUSE I/O disaster |
| NB03 — COCO from local SSD | A100 | ~14 | ~0.5 | ~7 CU | ✅ Expected with fix applied |
| NB04–07 | T4 | ~1.6 | ~2 | ~3 CU | Use T4, not A100 |

> Note: Previous session was on Colab Pro (~100 CU/month). Pro+ has ~500 CU/month but A100 costs ~13–15 CU/hr (not ~5 as previously logged). Recalibrate estimates accordingly.

**Efficiency targets going forward:**
- Debug on CPU/local → test on T4 → train on A100 only when code is confirmed
- Profile 1 epoch before full run — if GPU util < 80%, fix data pipeline first
- Always `runtime.unassign()` at end of training — idle A100 = 14 CU/hr burned for nothing
- Use L4 as the middle ground when T4 OOMs but A100 feels excessive

---

## 9. Key Rules For Next Agent

1. **Never point training data YAML to Drive.** Always extract to `/content/` (local SSD) first. Drive FUSE is ~5ms/file — catastrophic for 118K-image datasets.

2. **The ARGUS Drive folder is NOT a git repo.** Clone from GitHub each session to `/content/Argus/`. Drive only holds large binary artifacts.

3. **Session disconnects are non-destructive** — NB01–03 outputs are safe on Drive. Just re-clone and re-run the pipeline.

4. **llama-cpp-python cannot be built on Colab Python 3.12.** Do not attempt. Download GGUF only. Build llama-cpp-python on Jetson.

5. **pyaudio cannot be built on Colab.** Skip it. Audio I/O is a Jetson concern only.

6. **albumentations v2** uses `size=(h,w)` keyword arg, `scale` max 1.0.

7. **ONNX export from PyTorch 2.x** requires `dynamo=False` for any model using `cudnn_grid_sampler` (RAFT-Stereo family).

8. **T4 is fine for NB04–07.** Switch to T4 to save compute units. Only NB01–03 benefit from A100.

9. **TensorRT conversion must happen on Jetson.** TRT engines are device-specific — you cannot convert on Colab and deploy to Jetson.

10. **Stereo calibration** hasn't been done yet. The AR0234 cameras need a checkerboard calibration session before accurate depth can be produced.

---

## 10. Git Commit History (key fixes)

```
e72f121  Fix git pull: use fetch+reset --hard instead of checkout+pull --ff-only
c8cd10b  Fix git pull blocking + NB06 llama-cpp-python
3322c6c  Fix NB05 paths + NB06 llama-cpp-python install
1eb8f1d  Fix NB05: remove pyaudio, handle no-mic in headless Colab
3c28c0a  Fix NB04 EasyOCR bbox unpacking for newer API versions
5c3eea1  Fix pipeline runner: accept T4, fix stale CRAFT refs, smart GPU check
```

---

*Last updated: 2026-05-27 | Session by Claude Sonnet*

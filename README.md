# ARGUS — AI-Powered Smart Glasses for the Visually Impaired

Edge-AI smart glasses providing real-time navigation, open-vocabulary object finding, and scene understanding for visually impaired users. Validated locally on PC, deployed on the **NVIDIA Jetson Orin Nano Super (8 GB)**.

> **Architecture (post Phase-0):** ARGUS uses a **Two-Speed Vision-Language** design with **Gemma 4 E2B** (reasoning) + **YOLO-World** (grounding). YOLO-World replaces LocateAnything-3B, which Phase-0 found undeployable on the Orin Nano. Full design: [`docs/ARGUS_Final_Pipeline.pdf`](docs/ARGUS_Final_Pipeline.pdf) (with diagrams) and [`docs/ARGUS_Final_Pipeline.docx`](docs/ARGUS_Final_Pipeline.docx).

---

## Two-Speed Architecture

ARGUS runs at two speeds to stay within the Jetson's 8 GB / power budget. All models are **resident simultaneously** (~4.8–5.5 GB total) — no model swapping.

### ⚡ Fast Path — always on, low latency, non-ML safety
- **Stereo depth** (AR0234 pair, RAFT-Stereo / SGBM) → obstacle distance
- **Geometric safety reflex** → continuous, rule-based hazard detection (no ML)
- **SLAM** (OpenVINS / ORB-SLAM3) → pose tracking

### 🧠 Slow Path — event-driven, on demand
1. Wake word (openWakeWord) → faster-whisper (tiny, INT8) transcribes the query
2. Wide-camera frame passes the **mandatory privacy gate** (SCRFD + CRAFT face/text blur)
3. **Gemma 4 E2B** (multimodal, llama.cpp) reasons over frame + query
4. For object finding it calls **`find_object(name)`** → **YOLO-World** returns a box (~160 ms under TensorRT)
5. Box centre is sampled against the depth map → 3D position → **Piper** speaks the guidance

> *"Your keys are on the table to your right, about one metre ahead."*

### Why the change from the old pipeline?
The previous design was a heavy *specialist ensemble* (RAFT-Stereo + SegFormer + YOLOv8 + Phi-3.5 + separate face/text passes) run simultaneously — hard to keep in sync and heavy on the Jetson. Phase-0 confirmed Gemma 4 E2B is deployable but LocateAnything-3B is **not** (BF16-only, no GGUF/TensorRT path, Hopper/Blackwell-only kernels). Substituting **YOLO-World** preserves open-vocabulary object finding, cuts grounding latency by ~10×, and relaxes the memory budget.

---

## Workflow: PC test → Jetson

1. **PC Test Phase (Option C)** — on a local Windows PC + GTX 1060 6GB, validate cameras, YOLO-World detection, the speech stack, and ONNX export. See **[SETUP_LOCAL.md](SETUP_LOCAL.md)** and run **[`argus_pc_test.ipynb`](argus_pc_test.ipynb)**. *No Colab; no VLM on the PC.*
2. **Jetson Deployment** — JetPack 6, build TensorRT engines on-device (`trtexec --fp16`), launch Gemma 4 E2B with vision via native llama.cpp.

> TensorRT engines are **device-specific** — built on the Jetson, not cross-compiled. Colab/PC produce ONNX only.

---

## Hardware

### Compute
- **NVIDIA Jetson Orin Nano Super (8GB)** — deployment target (Ampere GPU, CUDA arch 87)
- **PC test:** Windows + GTX 1060 6GB (Pascal 6.1, FP32 only — no Tensor Cores)

### Cameras
- **2× Arducam AR0234** (2.3MP global shutter, USB 3.0) — stereo depth pair
- **1× Arducam IMX477P** (12MP, USB 3.0, M12 wide lens) — wide scene camera

### Audio
- USB microphone (wake word + speech commands)
- Bone-conduction headphones / speaker (TTS output)

---

## Component Stack

| Role | Model / Method | Loop | Deployment |
|---|---|---|---|
| Reasoning agent | **Gemma 4 E2B** (multimodal) | Slow | llama.cpp (GGUF, INT4) |
| Grounding tool | **YOLO-World** | Slow | TensorRT (INT8/FP16) |
| Stereo depth | RAFT-Stereo / SGBM | Fast | TensorRT |
| Obstacle & hazard | Geometric rules | Fast | Native code |
| SLAM | OpenVINS / ORB-SLAM3 | Fast | Native code |
| Privacy gate | SCRFD + CRAFT | Slow | TensorRT |
| Wake word | openWakeWord | — | CPU |
| Speech-to-text | faster-whisper (tiny) | — | CPU (INT8) |
| Text-to-speech | Piper | — | CPU |

---

## Repository Structure

```
argus/                  # Jetson runtime package (the two-speed system)
  config.py cameras.py depth.py safety.py privacy.py grounding.py
  agent.py speech.py trt_runner.py orchestrator.py selftest.py __main__.py
scripts/
  setup_jetson.sh       # Jetson dependency installer (run on device)
  run_llama_server.sh   # launch Gemma 4 E2B vision server
  build_engines.sh      # ONNX -> TensorRT (on device)
  calibrate_stereo.py   # AR0234 stereo calibration
config/argus.yaml       # runtime config (seed template)
requirements-jetson.txt # Jetson Python deps
setup.py                # installs the `argus` package (python -m argus)
setup_pc.bat            # PC-test env installer (Windows only)
argus_pc_test.ipynb     # PC validation notebook (cameras, YOLO-World, speech, ONNX export)
SETUP_LOCAL.md          # Step-by-step local PC setup guide
AGENT_HANDOFF.md        # Development history, lessons learned, efficiency rules
docs/
  PROJECT_OVERVIEW.md       # one-page orientation + repo map
  ARCHITECTURE.md           # module-level design + diagrams
  JETSON_DEPLOYMENT.md      # full on-device bring-up
  HARDWARE.md               # wearable assembly + wiring
  JETSON_CLAUDE_PROMPT.md   # kickoff prompt for Claude Code on the Jetson
  ARGUS_Final_Pipeline.pdf  # authoritative design doc (with diagrams)
  ARGUS_Final_Pipeline.docx # text transcription
```

### Quick start
- **Jetson (deployment):** clone, then `./scripts/setup_jetson.sh` → follow [docs/JETSON_DEPLOYMENT.md](docs/JETSON_DEPLOYMENT.md). To have Claude Code drive bring-up, use [docs/JETSON_CLAUDE_PROMPT.md](docs/JETSON_CLAUDE_PROMPT.md).
- **PC (validation):** `setup_pc.bat` → open `argus_pc_test.ipynb` (see [SETUP_LOCAL.md](SETUP_LOCAL.md)).
- Runtime CLI: `python -m argus selftest` · `python -m argus run` · `python -m argus query "what is in front of me?"`

### Branches
- **`main`** — current **Two-Speed / YOLO-World** work (PC-test-first, then Jetson)
- **`legacy-specialist-pipeline`** — the previous specialist-ensemble pipeline (RAFT-Stereo, SegFormer, YOLOv8, Phi-3.5, EasyOCR, the Colab orchestrator and `src/` runtime), fully preserved with complete history

The old files were removed from `main` for a clean slate; they remain intact on `legacy-specialist-pipeline`.

---

## Compute Resources & Efficiency Rules

| Resource | Spec | Role |
|---|---|---|
| GTX 1060 (local) | 6 GB, ~150 GB disk | Code authoring, debugging, data prep, calibration, ONNX validation |
| Colab T4 | 15 GB | First-pass testing; export, quantisation, inference-only |
| Colab L4 | 22.5 GB | Mid-tier fine-tuning where T4 is memory-constrained |
| Colab A100 | 40 GB | Heavier fine-tuning — only after a 1-epoch T4 profiling pass |
| Google Drive | Persistent | Model artefacts & dataset archives only |

Non-negotiable rules: never train/read datasets directly from Drive (stage locally first); always use mixed precision **on Colab** (FP32 locally on Pascal); profile one epoch before scaling; checkpoint to persistent storage periodically; release the accelerator on completion; apply early stopping. All ONNX exports use `dynamo=False`. See [AGENT_HANDOFF.md](AGENT_HANDOFF.md).

---

## Status

🚧 In active development — **PC test phase**.

- ✅ Phase-0 feasibility closed → Two-Speed / YOLO-World architecture locked
- ✅ Legacy pipeline preserved (`legacy-specialist-pipeline` branch)
- 🔄 PC test: cameras, YOLO-World, speech, ONNX export
- ⏭️ Jetson deployment: TensorRT engines, Gemma 4 E2B slow path

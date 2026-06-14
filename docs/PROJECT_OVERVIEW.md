# ARGUS — Project Overview & Setup

A single-page orientation to the whole project: what it is, how the repo is laid
out, and where to go next depending on what you're doing.

---

## What ARGUS is

AI-powered smart glasses that help a visually impaired user navigate and
understand their surroundings. Three cameras on a 3D-printed frame feed an
**NVIDIA Jetson Orin Nano Super**, which runs a **Two-Speed Vision-Language**
system:

- **Fast path (always on, non-ML):** stereo depth + a geometric safety reflex →
  immediate spoken warnings about obstacles and drop-offs.
- **Slow path (on demand):** "Hey ARGUS, find my keys" → speech-to-text →
  privacy gate (blur faces) → **Gemma 4 E2B** multimodal agent → **YOLO-World**
  open-vocabulary grounding → fuse with depth → spoken guidance.

The reasoning agent (Gemma 4) and the grounder (YOLO-World) both run on-device.
The design rationale is in [`ARGUS_Final_Pipeline.pdf`](ARGUS_Final_Pipeline.pdf).

---

## Repository layout

```
ARGUS/
├── README.md                  # top-level intro + branch structure
├── PROJECT_OVERVIEW.md ........ (this file lives in docs/)
├── argus/                     # the runtime package (two-speed system)
│   ├── config.py              #   dataclass config + YAML loader
│   ├── cameras.py             #   3-camera rig (stereo + wide)
│   ├── depth.py               #   SGBM / RAFT-Stereo depth
│   ├── safety.py              #   non-ML geometric safety reflex
│   ├── privacy.py             #   mandatory face-blur gate
│   ├── grounding.py           #   YOLO-World find_object
│   ├── agent.py               #   Gemma 4 E2B llama.cpp client + tool contract
│   ├── speech.py              #   wake word, Whisper STT, Piper TTS
│   ├── trt_runner.py          #   TensorRT engine runner (Jetson)
│   ├── orchestrator.py        #   the two-speed nervous system
│   ├── selftest.py            #   bring-up checks
│   └── __main__.py            #   CLI: run / query / selftest
├── scripts/
│   ├── setup_jetson.sh        # Jetson dependency installer (run on device)
│   ├── run_llama_server.sh    # launch Gemma 4 vision server
│   ├── build_engines.sh       # ONNX -> TensorRT (on device)
│   └── calibrate_stereo.py    # AR0234 stereo calibration
├── config/argus.yaml          # runtime config (seed; live copy on device)
├── requirements-jetson.txt    # Jetson Python deps
├── setup.py                   # installs the `argus` package
├── setup_pc.bat               # PC-test env installer (Windows only)
├── argus_pc_test.ipynb        # PC validation notebook (GTX 1060)
├── SETUP_LOCAL.md             # PC test setup guide
└── docs/
    ├── ARCHITECTURE.md
    ├── JETSON_DEPLOYMENT.md
    ├── HARDWARE.md
    ├── JETSON_CLAUDE_PROMPT.md   # kickoff prompt for Claude Code on the Jetson
    ├── ARGUS_Final_Pipeline.pdf  # authoritative design doc
    └── ARGUS_Final_Pipeline.docx
```

---

## Where to go next

| If you want to… | Go to |
|---|---|
| Understand the design | [ARCHITECTURE.md](ARCHITECTURE.md) + the PDF |
| Test on the PC (1060) | [../SETUP_LOCAL.md](../SETUP_LOCAL.md) + `argus_pc_test.ipynb` |
| Build the wearable | [HARDWARE.md](HARDWARE.md) |
| Deploy on the Jetson | [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md) |
| Get Claude to drive Jetson bring-up | [JETSON_CLAUDE_PROMPT.md](JETSON_CLAUDE_PROMPT.md) |
| See project history / lessons | [../AGENT_HANDOFF.md](../AGENT_HANDOFF.md) |

---

## Two ways to install

- **PC (Windows, GTX 1060)** — validation only:
  `setup_pc.bat`  → then `argus_pc_test.ipynb`.
- **Jetson (Ubuntu ARM64)** — the real deployment:
  `./scripts/setup_jetson.sh`  → see [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md).

> The `.bat` is Windows-only and cannot run on the Jetson. The Jetson installer
> is `scripts/setup_jetson.sh` (Linux/ARM64).

---

## Branches

- **`main`** — current Two-Speed / YOLO-World work (this codebase).
- **`legacy-specialist-pipeline`** — the previous specialist-ensemble pipeline
  (RAFT-Stereo + SegFormer + YOLOv8 + Phi-3.5 + EasyOCR + Colab orchestrator),
  preserved with full history.

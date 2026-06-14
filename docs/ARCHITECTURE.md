# ARGUS — Architecture

ARGUS is AI-powered smart glasses for the visually impaired. It runs a
**Two-Speed Vision-Language** system on an **NVIDIA Jetson Orin Nano Super (8 GB)**.

For the formal design rationale (incl. the Phase-0 feasibility findings that led
to YOLO-World replacing LocateAnything-3B), see
[`ARGUS_Final_Pipeline.pdf`](ARGUS_Final_Pipeline.pdf).

---

## The two loops

```
                          ┌──────────────────────────────────────────┐
                          │              CameraRig                    │
                          │  AR0234 L  AR0234 R       IMX477P wide     │
                          └─────┬───────┬──────────────────┬─────────┘
                                │ stereo pair              │ wide frame
              ┌─────────────────▼──────────┐    ┌──────────▼──────────────────┐
   FAST LOOP  │ DepthEstimator (SGBM/RAFT)  │    │  (used by the slow loop)     │
  (always on) │ → SafetyReflex (geometric)  │    │                              │
              │ → speak DANGER immediately  │    │                              │
              └─────────────────────────────┘    │                              │
                                                  │                              │
   SLOW LOOP  wake word → record → Whisper STT ──▶│  PrivacyGate (blur faces)    │
 (event-driven)                                   │        │ MANDATORY            │
                                                  │        ▼                     │
                                                  │  GemmaAgent (Gemma 4 E2B)    │
                                                  │        │                     │
                                                  │  find_object(name)?          │
                                                  │        ▼                     │
                                                  │  Grounder (YOLO-World) ──┐   │
                                                  │        │  box             │   │
                                                  │  fuse with depth map ◀───┘   │
                                                  │        ▼                     │
                                                  │  Gemma final answer          │
                                                  │        ▼                     │
                                                  │  Piper TTS speaks            │
                                                  └──────────────────────────────┘
```

### Fast loop — `argus/safety.py`, `argus/depth.py`
Continuously: capture stereo → depth map → **geometric** rules (no ML) → if an
obstacle is within the danger distance, or a floor drop-off is detected, speak a
short urgent warning immediately. Deterministic and always available, even while
the slow loop is busy talking to the agent.

### Slow loop — `argus/orchestrator.py`
Triggered by the wake word. Records the query, transcribes it, grabs the wide
frame, **passes it through the privacy gate (hard precondition)**, then asks
Gemma 4. If Gemma wants to locate something it calls `find_object(name)`; the
runtime runs YOLO-World, fuses the box centre with the depth map for a distance +
direction, and feeds that back to Gemma for the final spoken answer.

---

## Modules

| File | Responsibility |
|---|---|
| `argus/config.py` | Dataclass config + YAML loader (`/opt/argus/config/argus.yaml`) |
| `argus/cameras.py` | `CameraRig`: 3 cameras, threaded wide-frame, synced stereo grab |
| `argus/depth.py` | `DepthEstimator`: SGBM (CPU) or RAFT-Stereo (TensorRT) → metric depth |
| `argus/safety.py` | `SafetyReflex`: non-ML obstacle + drop-off detection |
| `argus/privacy.py` | `PrivacyGate`: SCRFD face blur — mandatory before the agent |
| `argus/grounding.py` | `Grounder`: YOLO-World open-vocab `find_object` |
| `argus/agent.py` | `GemmaAgent`: llama.cpp client, system prompt, tool contract |
| `argus/speech.py` | `WakeWord`, `Transcriber` (Whisper), `Speaker` (Piper), mic I/O |
| `argus/trt_runner.py` | Minimal TensorRT engine runner (Jetson only) |
| `argus/orchestrator.py` | Two-speed orchestration (the nervous system) |
| `argus/selftest.py` | Bring-up checks (deps, CUDA, models, cameras, server) |
| `argus/__main__.py` | CLI: `run`, `query`, `selftest` |

---

## Memory budget (Orin Nano 8 GB)

All components are resident simultaneously — no model swapping:

| Component | Precision | ~Memory |
|---|---|---|
| Gemma 4 E2B | INT4 GGUF | 2.0–2.5 GB |
| YOLO-World | FP16/INT8 TRT | 0.3–0.5 GB |
| Stereo depth | INT8 TRT / CPU | 0.2 GB |
| SLAM (future) | — | 0.5 GB |
| Privacy gate | INT8 | 0.2 GB |
| Speech (CPU) | INT8 | 0.1 GB |
| OS/CUDA/buffers | — | 1.5 GB |
| **Total** | | **~4.8–5.5 GB** (headroom within 8 GB) |

---

## Design rules carried from the project (do not break)

1. ONNX exports use `dynamo=False` (PyTorch 2.x dynamo fails on some vision ops).
2. TensorRT engines are **device-specific** — built on the Jetson, never cross-compiled.
3. The privacy gate is a **hard precondition** of every agent call.
4. The fast safety loop is **non-ML** and runs independently of the agent.
5. Speech (wake/STT/TTS) runs on **CPU**.

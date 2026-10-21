# ARGUS implementation status

Last audited: 2026-10-21. This file is the implementation source of truth.

Status meanings: **done** means implemented and verified for its stated scope;
**partial** means useful code/evidence exists but acceptance is incomplete;
**broken** means the current path cannot be relied upon; **not started** means no
implementation exists.

| Component | Status | Verified reality / next gate |
|---|---|---|
| Jetson environment | partial | L4T R36.5.2, MAXN_SUPER. Over-current throttling (oc3) occurs under heavy GPU load on the original PSU; `doctor` and benches record the counter. `jetson_clocks --show` unverified (needs the user's sudo) |
| Rig doctor | done | `python3 -m argus doctor` (+ `--snap`, `--skew-test`, `--bandwidth`, `--focus`): all green with 3 cameras on USB3 (`reports/doctor-2026-10-18-2353.json`). Probes reopen a camera that drops off USB |
| Camera identity/transforms | done (re-verify on recable) | Roles bound by hub port: left 2-1.1, right 2-1.2, wide 2-1.4 (2026-10-21, by parallax). Rotations 0 (worn). Stereo 60 fps (`reports/usb-bandwidth-2026-10-02-2112.json`) |
| Stereo mount alignment | partial | Printed bridge; guided `argus align` brought vertical from 118 px to ~±5-15 px, roll ~1 deg. No clean final measurement (last one discarded: rig was being handled) |
| Stereo calibration tool | partial | Spoken `--coach`; held-out gate (mono RMS <= 0.5, stereo <= 0.7, row mean < 0.5 / max < 1 px, tape baseline +/-10%, >= 15 views, >= 70% coverage); k3 fixed. **No calibration file yet**: needs the user (square size, baseline, flat board) |
| GPU stereo depth | partial | CUDA SAD default (7.6 ms median 480x300/64); VPI SGM selectable (13.0 ms), confidence -> UNKNOWN (`reports/depth-backends-2026-10-21.json`). Metric accuracy blocked on calibration |
| Corridor hazards | partial | `argus/hazards.py`: ground plane, body/head bands, head-level rule, drop-off, head-motion gate (deg/s), 3-frame persistence, refractory, hysteresis; 5 ms/tick; 12 ray-cast tests. Speech only with calibration. Test walk pending |
| Approaching person/vehicle | partial | YOLO11n TRT (3.3 ms GPU, 14.4 ms end to end) + ByteTrack-style tracker + range Kalman + TTC hysteresis; 10 tests; live person detected. Speech only with calibration |
| Legacy reflex / TTC zones | partial | `argus/safety.py`; now keeps only the approach-zone rule when the corridor engine is on |
| Fast-loop timing | **broken under full stack** | Depth heartbeat gaps 0.26-0.73 s several times a minute and skew drops 12-20 ms with Gemma + Whisper + demo view + recording (`reports/sessions/2026-10-21-124310.jsonl`). Root cause not found. Must fix before walking tests |
| Speech arbiter | done (unit) | P0-P4, preemption < 50 ms in test, answer resume / "repeat?" offer, user-speech hold, depth watchdog earcon (rate-limited). Legacy priority suite fixed and green |
| Wake word | partial | "Argus" / "Hey Argus" via VAD + Whisper tiny with a glossary hint (9/9 synthetic, 0.78 s). Real voice: the name recognised every time; bare "Argus" -> "Yes?". False wakes/hour not measured. Trained openWakeWord model not built (`training/hey_argus/`) |
| VAD / STT | partial | Silero endpointing (0.7 s, 15 s cap) works live; Whisper tiny English only; mishearings seen ("Read the" -> "I'm the") |
| Conversation layer | partial | `argus/dialogue.py`: rule routing + Gemma JSON routing (13/14, 0.99 s), context, follow-ups without the wake word, exit, filler, persona. Live voice: chat 1.6 s, describe 2.4-3.0 s. Reading/seat flows not yet voice-tested end to end |
| Reading / OCR | partial | RapidOCR CPU; locate document -> CLAHE crop -> OCR (1.5 s); light/glare/blur-aware coach; diagnose and retry. Synthetic sign perfect; real book cover failed on optics (fixed-focus lens, backlight). Needs `doctor --focus` + CER test. Bangla not evaluated |
| Sensitive-text privacy | done (unit) | Luhn cards, BD mobiles, OTP/PIN codes masked before logs, screen and Gemma; demo face blur display-only |
| Face privacy gate | partial | InsightFace, threshold 0.25 after a 0.27 miss; fail-closed. Adversarial set still needed |
| Gemma 4 agent | partial | QAT UD-Q4_K_XL + vision-only mmproj on GPU, 280-token images; describe ~2.2 s; text-only routing/answers. Qwen3.5-2B bake-off not done |
| YOLO-World grounding | partial | TRT FP16 + 401-label precomputed vocabulary (no torch at runtime); optional conf override (reading uses 0.12) |
| Seat guidance | partial | Closed loop on YOLO11n chair/couch/bench; corridor whitelist; direction only until calibration. Live voice test pending (seen when the user stood up) |
| SLAM / navigation | partial | Stereo VO + corridor guidance exist with synthetic tests; off until calibration |
| Demo view / recording | done | `run --demo [--record]`: POV with display face blur, captions, hazard banner, insets, chips; x264 via ffmpeg thread (1.5 ms/frame); TTS audio muxed; closes only on Esc x2 or "shut down" |
| Session logger / study metrics | done | JSONL with masking; `scripts/summarise_session.py` (alarms/min, lead, task times, latency percentiles, precision/recall vs annotated truth) |
| Memory budget | partial | Per-component RSS measured; projected stack ~4.6 GB; 10-minute soak not done |
| Headless service/soak/fault tests | not started | No systemd unit or long soak |

## Reproducibility pins

Python additions (2026-10-21, `pip install --no-deps` where noted): `rapidocr_onnxruntime==1.4.4` (--no-deps), `pyclipper`, `shapely`. `faster-whisper` bundles the Silero VAD used by `argus/vad.py`. OCR runs on onnxruntime 1.23.2 CPUExecutionProvider (no GPU provider in this build).

### Platform

- Jetson Linux: `nvidia-l4t-core 36.5.2-20260716114719`.
- TensorRT: `10.3.0.30-1+cuda12.5`; `trtexec` reports v10.3.0.
- Python: 3.10 on ARM64.
- NVIDIA Torch: `2.5.0a0+872d972e41.nv24.8`.
- Torchvision: `0.20.0a0+afc54f7`.
- llama.cpp: `ef8268feee28ae943958049bf3bbab4bda99c0ea`, CUDA arch 87.

### Device artifact SHA-256

| Artifact | SHA-256 |
|---|---|
| `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` (production) | `e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889` |
| `mmproj-gemma4-e2b-vision-f16.gguf` (production, derived by `scripts/strip_mmproj_audio.py`) | `366316f696d52b65e33d69cdc55d8a1b3918164708b18df14949b6556b5559c9` |
| `clip/vocab_embeddings.npy` (401 labels, `scripts/build_vocab_embeddings.py`) | `b2db59b8ee6b20e36b5196558a0f57007f24d4ce3eb5fd1955b9eaa2bd0b7da6` |
| `gemma-4-E2B-it-Q4_K_M.gguf` (previous) | `9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d` |
| `mmproj-gemma4-e2b-f16.gguf` | `140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa` |
| `ocr/en_PP-OCRv3_rec_infer.onnx` (SWHL/RapidOCR; evaluated, not default) | `ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea` |
| `yolo11n_416x640_fp16.engine` (movers; trtexec FP16 on device) | `879c51fc923af746e2a2ad85197ad6ec77b1a6e47f296c67a8f95299bcc029a4` |
| `yolo11n_416x640.onnx` (Ultralytics 8.4.67 export, opset 17) | `8d338f38565b2787ea8b7e8661b81164268eddcb545ef3b2f47d2a983e72b5d7` |
| `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `yolov8s-worldv2.pt` | `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792` |
| `yoloworld_640.onnx` | `8b407bb9f5a206d8290fb20d1a6a7d11bc8788da269ec9a583b786290dba4545` |
| `yoloworld_640_fp16.engine` | `8d6b7649ef87f6e280b3e84f492b7b8d910e4d5318f6e31e64aa0659c9fa1912` |
| `ViT-B-32.pt` | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| `yoloworld_runtime_text_640.onnx` | `7f69826f578b66cc057b4eb81659456145ff3962e295981a51682b318f3123fb` |
| `yoloworld_runtime_text_640_fp16.engine` | `7f0cf0a82c0bc5ee713eb91b7085219263160ef806282a3617748abc13b94bfd` |
| Piper ONNX | `5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f` |
| Piper JSON | `efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0` |

The legacy engine above is fixed COCO-80 and is not used. The runtime-text engine
was exported and built on this Jetson with TensorRT 10.3, FP16, and a 2048 MiB
workspace. See `reports/grounding-runtime-text-2026-08-14.json`.

### Python dependencies

`requirements-jetson.txt` contains exact pins for the audited environment.
PyCUDA `2024.1.2` was compiled on-device and initialized one Orin device. Its
transitive packages are pinned too. Do not substitute a generic PyPI Torch build.

## Latest environment report — 2026-08-14

Command: `python3 -m argus --config config/argus.yaml baseline --output
reports/environment-baseline-2026-08-14.json`.

- 24 required checks passed; production readiness remains false.
- A real CUDA matrix workload completed in 0.208 s and drove `tegrastats` GR3D
  to 99%, proving Jetson GPU execution rather than inferring it from imports.
- Both B0495 stereo cameras, the B0459 wide camera, and USB capture/playback
  audio enumerate. All cameras are attached at 5 Gbit/s behind the same USB 3
  hub/controller; sustained bandwidth is deferred to the camera feature.
- Shared memory: 7,976,845,312 bytes; six zram swap devices active.
- All audited model and engine hashes matched.
- Required failures: `jetson_clocks --show` requires interactive sudo and stereo
  calibration is absent. MAXN_SUPER, CUDA stereo, and GPU grounding pass.
- The agent cannot verify privileged locked-clock state without the user's sudo;
  it must not request, store, or bypass that password.

## Test evidence

- 2026-10-21: `python3 -m pytest -q tests/` -> **295 passed** in ~22 s on the
  Jetson (27 test files, including the formerly hanging speech-priority suite,
  VPI, CUDA and TensorRT-free unit tests). Every new module has tests:
  `test_doctor`, `test_timecode`, `test_align`, `test_calibration_gate`,
  `test_reading`, `test_arbiter`, `test_vad`, `test_hazards`, `test_movers`,
  `test_seat`, `test_dialogue`, `test_wake_stt`, `test_vocab`,
  `test_vpi_stereo`, `test_session_summary`.
- Device findings and measurements: `docs/FINDINGS_2026-10.md`.

## Latest GPU depth report — 2026-08-14

`reports/cuda-stereo-depth-2026-08-14.json` records 60 live-frame runs. Warm
latency was 12.7 ms median and 14.1 ms p95 in 15W mode; sampled GR3D peaked at 95%, with RAM
between 3,985 and 4,114 MB. Pair skew was 1.03 ms median but the initial pair was
369 ms and therefore outside the production 12 ms gate. No calibration exists,
so the report is GPU/performance evidence, not metric-distance acceptance.

## Early end-to-end demo — 2026-08-14

`reports/early-e2e-demo-2026-08-14.json` records the positive manual query
“Find the monitor.” The fast CUDA depth loop stayed alive, the 368 ms initial
pair was rejected, privacy ran first on CPU, TensorRT found the monitor at center,
Gemma produced the final answer, and Piper/Pulse playback completed. Slow-path
stages were 0.118 s privacy, 4.269 s visual Gemma, 15.545 s cold grounding, and
1.141 s final Gemma. Peak RAM was 7,376 MB with substantial swap, so co-residency
and cold-start latency remain explicit blockers to wearable acceptance.

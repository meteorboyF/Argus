# ARGUS implementation status

Last audited: 2026-10-02. This file is the implementation source of truth.

Status meanings: **done** means implemented and verified for its stated scope;
**partial** means useful code/evidence exists but acceptance is incomplete;
**broken** means the current path cannot be relied upon; **not started** means no
implementation exists.

| Component | Status | Verified reality / next gate |
|---|---|---|
| Jetson environment baseline | partial | Refreshed report: 24 pass/2 fail. MAXN_SUPER is active; privileged `jetson_clocks --show` remains unverified and calibration is absent |
| Camera identity/transforms | partial | Each camera has its own capture thread with real arrival timestamps; roles bind to the hub port so a camera that falls back to USB 2 (`1-2.x` instead of `2-1.x`) keeps its role and is flagged. Brackets now mount both AR0234s upright, rotations set to 0. On 2026-09-29 hub ports 2 and 4 linked at 480 Mbit/s (10 fps, multi-second stalls); reseat needed before calibration |
| Stereo calibration tool | partial | Threaded capture, pose diversity and hard RMS/vertical gates; accepts the physical 8x8 folding board (7x7 inner corners). No deployed calibration yet; 0.5–3 m validation pending |
| Calibration drift monitor | partial | 12 synthetic tests pass; needs real calibrated scenes |
| GPU stereo depth | partial | Deterministic CUDA SAD backend is production default and never falls back to CPU. Live median 12.7 ms, p95 14.1 ms, GR3D peak 95%. Synthetic shift passed; physical metric accuracy is blocked on calibration |
| CPU SGBM | diagnostic only | Implemented; not permitted as production depth |
| Safety rules | partial | Static obstacle/drop rules plus the temporal approach rule (per-zone closing speed and time-to-collision); 77 unit tests pass. Uncalibrated warnings stay suppressed |
| Incoming vehicle warning | partial | Eye-level band split into left/center/right zones; closing speed from a 0.8 s linear fit, WARN at 3 s TTC, DANGER at 1.5 s. Synthetic 5 m/s approach test passes; real-scene validation blocked on calibration |
| Speech pipeline | partial | Six priority/preemption tests now pass. Piper completed through PulseAudio USB without PortAudio underruns. Wake word/STT microphone loop is not yet accepted; wake phrase remains `hey_jarvis` |
| YOLO-World TensorRT | partial | Production FP16 runtime-label engine. 401-label vocabulary embedded offline (parity 0.0); warm-up no longer loads torch/CLIP: RSS 1672 -> 353 MB, load 11.6 -> 1.6 s (`reports/mem-budget-2026-10-02.json`). Out-of-vocabulary labels still load CLIP lazily |
| Ultralytics grounding | diagnostic only | `.pt` prototype exists; prohibited production fallback |
| Face privacy gate | partial | InsightFace blur is required and fail-closed. 2026-10-02: a dark, side-on, edge-cropped face scored 0.27 and passed the 0.5 default unblurred; threshold now 0.25. Adversarial set still needed |
| Sensitive-text privacy | not started | No CRAFT/text flag or blur implementation |
| Gemma 4 agent | partial | 2026-10-02: QAT UD-Q4_K_XL + vision-only mmproj on GPU, `--image-max-tokens 560`, client sends ~280 tokens. Live `GemmaAgent.ask` 2.23 s / 1.91 s; QAT bench describe 2.18 s, find 1.66 s, read (556 tok) 3.54 s warm, all under over-current throttling (`reports/vlm-m3-qat-2026-10-02.json`). One live answer missed a visible blurred fan: accuracy needs the 30-frame bake-off |
| Agent tool protocol | partial | Explicit locate intent is deterministically forced through verified TensorRT grounding when Gemma ignores the prompt. Positive monitor query returned center; broader language tests remain |
| Wide-to-stereo projection | broken | Old proportional mapping disabled; direction only until calibrated projection exists |
| SLAM | partial | Stereo visual odometry in `argus/slam.py` (ORB stereo triangulation, LK tracking, PnP RANSAC) on its own thread; two synthetic tests pass with metric scale. Off until stereo calibration exists |
| Full two-speed integration | partial | `run --dashboard` shows cameras, depth, top-down map, status and the conversation on the monitor; questions can be typed or appended to a file. Verified live 2026-09-29. Wake/STT and memory hardening remain |
| Monitor dashboard | done | `argus/dashboard.py` + `argus/telemetry.py`: OpenCV window with three feeds, depth, corridor/SLAM map, status and speech log; verified at 1920x1080 on 2026-09-29 |
| Corridor navigation | partial | `argus/navigation.py` picks the widest free bearing from calibrated depth; three synthetic tests pass. Off until calibration exists |
| Rig doctor | partial | `python3 -m argus doctor` (+ `--snap`, `--skew-test`, `--bandwidth`) verified with one B0495: 29.9/59.8/79.5 fps at 30/60/80 on USB3. Full three-camera run pending reconnection |
| Headless service/soak/fault tests | not started | No service definition or sustained validation |

## Reproducibility pins

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

- `tests/test_cameras.py`: 9 passed.
- `tests/test_safety.py`: 9 passed.
- `tests/test_calib_health.py`: 12 passed.
- `tests/test_fail_closed.py`: 14 passed.
- `tests/test_diagnostics.py`: 4 passed.
- `tests/test_cuda_stereo.py`: 3 passed on the Jetson, including worker-thread CUDA.
- `tests/test_speech_priority.py`: 6 passed, including DANGER preemption.

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

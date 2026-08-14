# ARGUS implementation status

Last audited: 2026-08-14. This file is the implementation source of truth.

Status meanings: **done** means implemented and verified for its stated scope;
**partial** means useful code/evidence exists but acceptance is incomplete;
**broken** means the current path cannot be relied upon; **not started** means no
implementation exists.

| Component | Status | Verified reality / next gate |
|---|---|---|
| Jetson OS/power | partial | L4T R36.5.2 and MAXN_SUPER observed; `jetson_clocks` and full baseline report still required |
| Camera identity/transforms | partial | Stable USB-role code and 9 unit tests pass; delivered FPS/skew/reconnect require device test |
| Stereo calibration tool | partial | Substantial capture/solve code exists; no deployed calibration or 0.5–3 m validation |
| Calibration drift monitor | partial | 12 synthetic tests pass; needs real calibrated scenes |
| GPU stereo depth | broken | No RAFT ONNX/engine found; production refuses CPU fallback |
| CPU SGBM | diagnostic only | Implemented; not permitted as production depth |
| Safety rules | partial | 9 synthetic obstacle/drop tests pass; uncalibrated warnings suppressed; no approach/TTC validation |
| Incoming vehicle warning | not started | Requires temporal range/approach rules and controlled tests |
| Speech pipeline | broken | Components exist, but six-test priority suite hangs during shutdown; wake word is `hey_jarvis` and capture is fixed length |
| YOLO-World TensorRT | broken | Engine exists but runtime does not consume it; fixed/dynamic vocabulary contract unknown |
| Ultralytics grounding | diagnostic only | `.pt` prototype exists; prohibited production fallback |
| Face privacy gate | partial | InsightFace blur exists and is required; exception boundary is fail-closed; adversarial tests needed |
| Sensitive-text privacy | not started | No CRAFT/text flag or blur implementation |
| Gemma 4 agent | partial | CUDA multimodal inference verified historically; flash attention was off and full-stack latency remains unverified |
| Agent tool protocol | partial | Prompt/native parsing exists; no focused unit tests or verified TRT tool execution |
| Wide-to-stereo projection | broken | Old proportional mapping disabled; direction only until calibrated projection exists |
| SLAM | not started | No backend, IMU inventory, calibration, or pose contract |
| Full two-speed integration | broken | Skeleton exists; production startup intentionally fails until GPU paths are ready |
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
| `gemma-4-E2B-it-Q4_K_M.gguf` | `9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d` |
| `mmproj-gemma4-e2b-f16.gguf` | `140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa` |
| `yolov8s-worldv2.pt` | `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792` |
| `yoloworld_640.onnx` | `8b407bb9f5a206d8290fb20d1a6a7d11bc8788da269ec9a583b786290dba4545` |
| `yoloworld_640_fp16.engine` | `8d6b7649ef87f6e280b3e84f492b7b8d910e4d5318f6e31e64aa0659c9fa1912` |
| Piper ONNX | `5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f` |
| Piper JSON | `efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0` |

YOLO engine provenance: built on this Jetson from the listed ONNX with TensorRT
10.3 using FP16 and a 2048 MiB workspace. Its class/vocabulary contract and
runtime correctness are unverified, so the hash proves identity, not fitness.

### Python dependencies

`requirements-jetson.txt` contains exact pins for the audited environment.
`pycuda` is currently missing and is a blocker for the existing TensorRT runner.
Do not substitute a generic PyPI Torch build.

## Test evidence

- `tests/test_cameras.py`: 9 passed.
- `tests/test_safety.py`: 9 passed.
- `tests/test_calib_health.py`: 12 passed.
- `tests/test_fail_closed.py`: 7 passed.
- `tests/test_speech_priority.py`: hangs during execution/teardown; do not count
  its printed dots as a passing suite.

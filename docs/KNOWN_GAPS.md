# ARGUS — Known Gaps & Remaining Work

Honest inventory of what is **not finished** or **not yet verified on the
device**. Read this before assuming a feature works. Ordered by how much it
blocks a working demo.

_Last updated: 2026-07-13 (PC-side hardening pass; nothing below has run on a
physical Jetson yet)._

---

## A. Must be done ON the Jetson (cannot be done from a PC)

| # | Item | Where | Notes |
|---|---|---|---|
| A1 | **Install NVIDIA's torch wheel** and verify `torch.cuda.is_available()` | [JETSON_DEPLOYMENT.md §2a](JETSON_DEPLOYMENT.md) | pip default torch is CPU-only on ARM |
| A2 | **Pick + download the exact Gemma vision GGUF** (`Q4_K_M`) and its `mmproj` | §3 / `scripts/download_models.sh` | Config default filenames (`gemma-4-E2B-it-Q4_K_M.gguf`, `mmproj-gemma4-e2b-f16.gguf`) are placeholders — match them to the files you actually download, and confirm llama.cpp loads them with vision enabled |
| A3 | **Stereo calibration** with the real mounted cameras, then `--verify` against a tape measure. A printer-free target is now available via `scripts/display_calibration_board.py`; run the calibrator headless while moving the locked rig in front of the full-screen board. | [CALIBRATION.md](CALIBRATION.md) | Still not complete: EDID-scaled squares and recovered depth must be verified against a real ruler/tape measure before safety distances are trusted |
| ~~A4~~ | ~~Confirm camera discovery hints~~ — **done 2026-08-13.** This rig's driver reports Arducam module part numbers, not sensor names: `AR0234`/`IMX477` never appear in `v4l2-ctl --list-devices`. Actual strings are `B0495` (2.3MP, the AR0234 stereo pair, x2) and `B0459` (12MP, the IMX477P wide cam, x1). `argus.yaml` `stereo_name_hint`/`wide_name_hint` updated to match on both the live config and the repo template. | `argus/cameras.py` | If a different Arducam batch reports different strings, re-verify with `v4l2-ctl --list-devices` |
| A5 | **Build TensorRT engines** from ONNX (if using `raft_trt`) | `scripts/build_engines.sh` | Engines are device-specific; SGBM works without them |
| A6 | **Audio device mapping** — set `speech.input_device`/`output_device` after `python3 -m sounddevice` | `argus.yaml` | USB mic + bone-conduction enumeration is device-specific |
| A7 | **End-to-end latency + memory profiling** under `tegrastats` (fast-loop Hz, wake→answer time, RAM headroom). Partial improvement 2026-08-13: reasoning disabled, 256 px vision input, 48-token cap. Warm-cache benchmark reached 6.41 s, but the honest cold production preview was 25.6 s (12.7 s prompt/image + 12.8 s for 14 generated tokens), versus >120 s with the original reasoning/256-token profile. | — | Still not interactive enough; test Gemma dynamic image-token limits and shorter output, then measure full wake→STT→vision→TTS and all-components-resident memory peak |
| A8 | **Verify piper-tts API** — `PiperVoice.synthesize(text, wav_file)` signature has changed between piper versions; adjust `argus/speech.py::Speaker.speak` if the installed version differs | `argus/speech.py` | Selftest imports piper; speak a test phrase early |
| A9 | **Verify openWakeWord score keys** — `WakeWord.detected` matches score keys by prefix of `wake_model`; print `scores.keys()` once on device to confirm `hey_jarvis` matches | `argus/speech.py` | |

## B. Code that exists but is a stub / approximation (improve later)

| # | Item | Where | Notes |
|---|---|---|---|
| B1 | **YOLO-World TensorRT path not wired** — grounding runs the `.pt` via torch (works, ~interactive). The TRT engine built by `build_engines.sh` is not consumed by `argus/grounding.py`; open-vocab TRT needs the text-embedding head handled (bake a vocabulary at export, or export the prompt encoder separately) | `argus/grounding.py` | The ~160 ms target in the design assumes TRT; treat as an optimisation milestone |
| B2 | **Wide↔stereo fusion is proportional, not geometric** — `find_object` boxes come from the wide camera but depth from the stereo pair; the centre is sampled by scaling pixel coordinates. Good enough for "about a metre to your right"; a wide↔left extrinsic calibration would make it exact | `argus/orchestrator.py::_fuse_detection` | Requires a joint calibration routine (checkerboard seen by wide + left) |
| B3 | **No SLAM** — the design mentions OpenVINS/ORB-SLAM3 for pose; nothing is implemented. Not needed for obstacle warnings or find-object | design docs | Deliberate scope cut for first bring-up |
| B4 | **Text blur (CRAFT) not wired** — `privacy.enable_text_blur` exists but no CRAFT weights/inference; only faces are blurred | `argus/privacy.py` | Add CRAFT ONNX + region blur, then flip the config |
| B5 | **Custom "ARGUS" wake word missing** — using openWakeWord's built-in `hey_jarvis`. Train a custom model (openWakeWord provides a Colab recipe) and set `speech.wake_model` to the `.onnx` path | `argus.yaml` | |
| B6 | **Fixed 5 s query recording** — no VAD/end-of-speech detection; queries are always `record_seconds` long | `argus/speech.py::record` | Add energy/VAD trimming (e.g. webrtcvad or Silero) for snappier turns |
| B7 | **RAFT-Stereo ONNX not in repo** — `depth.backend: raft_trt` expects `raft_stereo_640x480.onnx` from the legacy Colab pipeline (`legacy-specialist-pipeline` branch). SGBM is the supported default | `argus/depth.py` | |
| B8 | **Safety reflex thresholds untuned** — `warn/danger` distances, `obstacle_percentile`, drop-off fraction are sensible defaults, not field-tested values. Tune while walking with the rig (`run --no-audio`, watch printed states) | `argus/safety.py`, `argus.yaml` | |
| B9 | **`find_all` hazard watchlist unused** — `Grounder.find_all` exists for a proactive hazard scan (e.g. "stairs", "car") but nothing calls it | `argus/grounding.py` | Possible slow-loop background job |

## C. Product/robustness work beyond first demo

- **Logging**: runtime prints to stdout; nothing writes `/opt/argus/logs`. Add
  rotating file logging + a `--log-level` flag.
- **Watchdog/systemd hardening**: auto-restart is configured in the sample
  units, but there's no health monitoring of the fast loop itself.
- **Battery/thermal management**: `nvpmodel -m 0` at 25 W drains a pack fast;
  profile whether 15 W mode still holds the fast loop at 10 Hz.
- **Multilingual STT/TTS**: Whisper is pinned to English; Piper voice is en_US.
- **Unit tests**: only PC smoke tests exist (safety reflex, agent parsing).
  Add pytest coverage for safety geometry and calibration maths (both pure
  functions, easily testable off-device).
- **PC test notebook drift**: `argus_pc_test.ipynb` predates this hardening
  pass — treat the `argus/` package as the source of truth where they disagree.

## D. Design rules that must NOT be broken (carried forward)

1. TensorRT engines are device-specific — build on the Jetson, never copy.
2. The privacy gate is a hard precondition of every agent call
   (`privacy.require_gate: true` enforces this — do not ship with it false).
3. The fast safety loop stays non-ML and independent of the agent.
4. ONNX exports use `dynamo=False`.
5. torch on Jetson = NVIDIA wheel, never pip default.
6. Speech (wake/STT/TTS) stays on CPU — the GPU belongs to Gemma + YOLO-World.

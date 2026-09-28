# ARGUS agent handoff

This file covers only the current Jetson runtime. Read `STATUS.md` first, then
`ARCHITECTURE.md` and `DECISION_LOG.md`. Material under `historical/` is evidence,
not current guidance.

## Baseline at handoff

- Target device: Jetson Orin Nano Super 8 GB, user `argus`.
- OS: L4T R36.5.2. The refreshed environment baseline has 24 passes and two
  required failures. MAXN_SUPER is active.
- Repository: `main`; audited-baseline cleanup is the current feature boundary.
- Native llama.cpp commit: `ef8268feee28ae943958049bf3bbab4bda99c0ea`.
- Gemma CUDA/multimodal execution was demonstrated historically, but must be
  remeasured with the full stack and flash attention enabled.
- Production grounding now uses a two-input FP16 TensorRT engine. One arbitrary
  label embedding is supplied at runtime; repeated labels are cached.
- GPU stereo now uses the deterministic CUDA SAD backend; no stereo calibration
  file exists, so metric warnings remain suppressed.
- The manual positive demo “Find the monitor” completed through privacy, Gemma,
  TensorRT grounding, a center-only result, and Piper/Pulse USB playback.
- 2026-09-29: `run --dashboard --no-mic --ask-file PATH` puts the whole runtime
  on the monitor and takes typed questions; a question round-trips in 1.8 s.
  Approach/TTC rules, stereo visual odometry and corridor guidance exist with
  synthetic tests (77 passing) and activate automatically once
  `/opt/argus/config/stereo_calib.npz` exists.
- Both AR0234s are now mounted upright on brackets; the physical calibration
  target is an 8x8 folding chess board (7x7 inner corners).

## Known-good evidence

- Camera transform and USB-role unit tests pass.
- Synthetic safety and calibration-health tests pass.
- Device logs show CUDA architecture 870 and successful privacy-gated Gemma
  image inference after the R36.5.2 upgrade.
- Models, ONNX, engine, Piper voice, and their SHA-256 hashes are listed in
  `STATUS.md`.
- A real CUDA workload drove GR3D to 99%; PyCUDA 2024.1.2 initializes the Orin.
- Both B0495 cameras, the B0459 camera, and USB input/output audio enumerate.

## Known blockers

1. The user must run `sudo nvpmodel -m 0 && sudo jetson_clocks`, then rerun the
   baseline so locked clocks can be verified. The agent cannot enter the password.
2. Capture and physically verify stereo calibration at 0.5, 1, 2, and 3 m.
3. Reduce cold grounding latency and 7.4 GB peak RAM / heavy swap under the full
   stack; cached TRT inference itself is fast.
4. Validate wake word + STT with the USB microphone; the six priority tests pass.
5. Implement sensitive-text privacy handling.
6. Calibrate wide-to-stereo geometry before returning object distance.
7. Reseat the stereo camera on hub port 2 and the wide camera on hub port 4 so
   they link at 5 Gbit/s; at 480 Mbit/s they deliver 10 fps with multi-second
   stalls, which the dashboard shows as skew drops and 0 fps.
8. Validate approach/TTC, SLAM and corridor guidance on real calibrated scenes.

## Working discipline

- Inspect the dirty worktree before editing; preserve unrelated user changes.
- Never describe an artifact's existence as proof its runtime path works.
- Verification must name the backend actually loaded and include latency,
  memory, and `tegrastats` GR3D evidence for GPU work.
- Keep production fail-closed. Diagnostic fallbacks must require an explicit
  non-production flag and must never be used in a wearable demo.
- After every feature update `STATUS.md`, this file, and `DECISION_LOG.md`, then
  commit and stop for user confirmation.

## What to do next

The dashboard build of 2026-09-29 is the current boundary. Next: fix the two
USB 2 links, run `scripts/calibrate_stereo.py` with the 7x7 board, validate at
0.5–3 m, then watch the approach rule, SLAM trail and corridor guidance come
alive on the dashboard. Keep cross-camera object distance disabled until its
separate calibration exists.

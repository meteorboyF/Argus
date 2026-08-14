# ARGUS agent handoff

This file covers only the current Jetson runtime. Read `STATUS.md` first, then
`ARCHITECTURE.md` and `DECISION_LOG.md`. Material under `historical/` is evidence,
not current guidance.

## Baseline at handoff

- Target device: Jetson Orin Nano Super 8 GB, user `argus`.
- OS: L4T R36.5.2. The refreshed environment baseline has 23 passes and three
  required failures; the device is currently in 15W, not MAXN_SUPER.
- Repository: `main`; audited-baseline cleanup is the current feature boundary.
- Native llama.cpp commit: `ef8268feee28ae943958049bf3bbab4bda99c0ea`.
- Gemma CUDA/multimodal execution was demonstrated historically, but must be
  remeasured with the full stack and flash attention enabled.
- Production grounding now uses a two-input FP16 TensorRT engine. One arbitrary
  label embedding is supplied at runtime; repeated labels are cached.
- GPU stereo now uses the deterministic CUDA SAD backend; no stereo calibration
  file exists, so metric warnings remain suppressed.

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
3. Validate grounding against positive physical objects; current device evidence
   proves GPU/runtime-label execution but the captured scene yielded null results.
4. Repair the hanging speech-priority tests.
5. Implement sensitive-text privacy handling.
6. Calibrate wide-to-stereo geometry before returning object distance.
7. Implement temporal approach/incoming-vehicle rules and later SLAM.

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

Feature 3 delivered CUDA stereo depth. After confirmation, prioritize the early
honest demo thread requested by the user: cameras -> CUDA depth -> one TensorRT
grounding query -> privacy-gated Gemma -> spoken reply. It may report direction
only: calibration and cross-camera geometry are still unverified, so distance
must remain omitted. Then return to physical calibration/hardening.

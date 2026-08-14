# ARGUS agent handoff

This file covers only the current Jetson runtime. Read `STATUS.md` first, then
`ARCHITECTURE.md` and `DECISION_LOG.md`. Material under `historical/` is evidence,
not current guidance.

## Baseline at handoff

- Target device: Jetson Orin Nano Super 8 GB, user `argus`.
- OS: L4T R36.5.2; power query reported MAXN_SUPER.
- Repository: `main`; audited-baseline cleanup is the current feature boundary.
- Native llama.cpp commit: `ef8268feee28ae943958049bf3bbab4bda99c0ea`.
- Gemma CUDA/multimodal execution was demonstrated historically, but must be
  remeasured with the full stack and flash attention enabled.
- A fixed FP16 YOLO-World engine exists but is not wired into runtime and may
  contain a baked vocabulary.
- No GPU stereo engine and no stereo calibration file exist.
- Production execution is intentionally fail-closed until those GPU paths work.

## Known-good evidence

- Camera transform and USB-role unit tests pass.
- Synthetic safety and calibration-health tests pass.
- Device logs show CUDA architecture 870 and successful privacy-gated Gemma
  image inference after the R36.5.2 upgrade.
- Models, ONNX, engine, Piper voice, and their SHA-256 hashes are listed in
  `STATUS.md`.

## Known blockers

1. Implement and verify TensorRT/CUDA stereo depth; CPU SGBM is diagnostic only.
2. Determine the YOLO-World engine's true vocabulary contract, then implement
   TensorRT grounding without a PyTorch fallback.
3. Repair the hanging speech-priority tests.
4. Capture and physically verify stereo calibration at 0.5, 1, 2, and 3 m.
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

Feature 1 is the environment baseline diagnostic. It must inventory camera and
audio devices outside the restricted development sandbox, prove JetPack-matched
CUDA/PyTorch, verify MAXN_SUPER plus `jetson_clocks`, record USB topology and
artifact hashes, and clearly fail any production requirement that is absent.
After its commit and confirmation, address GPU depth and the open-vocabulary
TensorRT grounding contract before building the early honest end-to-end demo.

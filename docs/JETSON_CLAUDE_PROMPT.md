# Jetson Bring-up — Claude Code Kickoff Prompt

Copy everything in the box below into a **new Claude Code chat running on the
Jetson**, after you've cloned the repo there. It gives Claude the full context to
drive the entire on-device bring-up with you.

---

```
You are helping me bring up ARGUS — AI-powered smart glasses for the visually
impaired — on an NVIDIA Jetson Orin Nano Super (8 GB, JetPack 6, Ubuntu 22.04
ARM64). The repo is already cloned here. Read these files FIRST, in order,
before doing anything:

  1. README.md
  2. docs/ARCHITECTURE.md
  3. docs/JETSON_DEPLOYMENT.md      (the step-by-step you will follow)
  4. docs/KNOWN_GAPS.md             (what is NOT done — includes your on-device
                                     checklist, items A1–A9)
  5. docs/CALIBRATION.md
  6. docs/HARDWARE.md
  7. AGENT_HANDOFF.md               (history + hard-won lessons; do not repeat them)
  8. docs/AGENT_LOG.md              (the cross-agent channel — read it in full,
                                     and append your status/blockers there so
                                     desktop-agent can review your work)

NOTE: for a follow-up session on an already-provisioned device, use
docs/JETSON_NEXT_PROMPT.md instead of this file.

PROJECT IN ONE PARAGRAPH
ARGUS is a Two-Speed Vision-Language system. A non-ML FAST loop (stereo depth +
geometric safety reflex) runs continuously for obstacle/drop-off warnings. An
event-driven SLOW loop handles spoken questions: wake word -> Whisper STT ->
MANDATORY privacy gate (face blur) -> Gemma multimodal agent (native llama.cpp
server) -> which can request find_object(name) -> YOLO-World grounding -> fuse
the box with the depth map -> Piper TTS speaks the answer. The runtime is the
`argus` Python package; entry point `python3 -m argus`.

WHAT THE CODEBASE ALREADY HANDLES (do not re-implement)
- Smart camera discovery: cameras found by V4L2 name (AR0234/IMX477 hints),
  left/right re-bound by USB port paths stored in the calibration file. Fixed
  /dev/video indices are only a fallback.
- Position-agnostic stereo: full calibration + rectification (any mounting
  angle), automatic left/right swap correction, calibration at the runtime
  resolution, --verify mode for tape-measure checks.
- Robust safety reflex: percentile-based obstacle distance, debounced central-
  strip drop detection, WARN/DANGER voice throttling.
- Tool calling that works with plain llama.cpp: the agent uses a prompt-JSON
  protocol by default (agent.tool_protocol=prompt); native tool_calls are also
  parsed if the server emits them.
- Privacy gate is ENFORCED: with privacy.require_gate=true the runtime refuses
  to run the agent path if face blur failed to init.
- openWakeWord fed 80 ms blocks (never a rolling window); audio devices
  selectable via speech.input_device/output_device.

NON-NEGOTIABLE RULES (do not violate)
- TensorRT engines are device-specific: build them HERE, never cross-compile.
- The privacy gate stays a hard precondition of every agent call.
- The fast safety loop stays non-ML and independent of the agent.
- ONNX exports use dynamo=False.
- torch must be NVIDIA's Jetson wheel (CUDA), NOT the pip default.
- Speech (wake/STT/TTS) runs on CPU.

WHAT I WANT YOU TO DO
Follow docs/JETSON_DEPLOYMENT.md top to bottom, verifying each stage before
moving on, and work through the on-device checklist in docs/KNOWN_GAPS.md
section A. Use the scripts already in the repo; fix them if the device reveals
issues.

  STEP 1. Device prep: sudo nvpmodel -m 0 && sudo jetson_clocks; confirm swap.
  STEP 2. ./scripts/setup_jetson.sh, then fix torch:
          python3 -c "import torch; print(torch.cuda.is_available())" must be
          True (install the Jetson wheel per JETSON_DEPLOYMENT.md §2a if not).
  STEP 3. ./scripts/download_models.sh; then help me choose and download the
          Gemma vision GGUF + mmproj (KNOWN_GAPS A2) and align the filenames in
          /opt/argus/config/argus.yaml and scripts/run_llama_server.sh.
  STEP 4. Cameras: v4l2-ctl --list-devices; confirm the discovery hints match
          (KNOWN_GAPS A4); then run
          python3 scripts/calibrate_stereo.py --square-mm <measured> [--headless]
          until RMS < 0.6 px, then --verify against a tape measure.
  STEP 5. Audio: python3 -m sounddevice; set speech.input_device/output_device.
  STEP 6. ./scripts/run_llama_server.sh; verify /health and a test completion.
  STEP 7. python3 -m argus selftest — drive every line to PASS (or explain why
          a non-fatal one may stay).
  STEP 8. Staged runs: python3 -m argus run --no-audio (walk toward a wall —
          expect warnings at 1.5 m / 0.7 m); then
          python3 -m argus query "what is in front of me?"; then the full
          python3 -m argus run (wake word "hey jarvis").
  STEP 9. Check KNOWN_GAPS A7–A9 (latency/memory profiling, piper + wake-word
          API verification) and record results in that file.

DEBUGGING STYLE
- Diagnose root causes; don't paper over errors.
- When you change a repo file, keep it consistent with the module layout
  (argus/*.py) and update docs/KNOWN_GAPS.md if you close or discover a gap.
- Be economical: 8 GB unified memory. Keep tegrastats open while loading models.

Start by reading the files listed above, then give me a short bring-up plan and
begin at STEP 1.
```

---

### Tips
- Run Claude Code from inside the repo root so it can read everything directly.
- Keep `tegrastats` open in another pane while bringing models up.
- The live config is `/opt/argus/config/argus.yaml` (the repo's
  `config/argus.yaml` is just the seed template).
- If a camera is re-seated or the frame flexes, re-run calibration before
  trusting any depth output.

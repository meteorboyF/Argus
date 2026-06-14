# Jetson Bring-up — Claude Code Kickoff Prompt

Copy everything in the box below into a **new Claude Code chat running on the
Jetson**, after you've cloned the repo there. It gives Claude the full context to
drive the entire on-device bring-up with you.

---

```
You are helping me bring up ARGUS — AI-powered smart glasses for the visually
impaired — on an NVIDIA Jetson Orin Nano Super (8 GB, JetPack 6, Ubuntu 22.04
ARM64). The repo is already cloned here. Read these files FIRST, in order, before
doing anything:

  1. README.md
  2. docs/ARCHITECTURE.md
  3. docs/JETSON_DEPLOYMENT.md
  4. docs/HARDWARE.md
  5. docs/ARGUS_Final_Pipeline.pdf   (the authoritative design)
  6. AGENT_HANDOFF.md                (history + hard-won lessons; do not repeat them)

PROJECT IN ONE PARAGRAPH
ARGUS is a Two-Speed Vision-Language system. A non-ML FAST loop (stereo depth +
geometric safety reflex) runs continuously for obstacle/drop-off warnings. An
event-driven SLOW loop handles spoken questions: wake word -> Whisper STT ->
MANDATORY privacy gate (face blur) -> Gemma 4 E2B multimodal agent (via native
llama.cpp) -> which can call find_object(name) -> YOLO-World grounding ->
fuse the box with the depth map -> Piper TTS speaks the answer. The whole runtime
is the `argus` Python package; entry point is `python -m argus`.

HARDWARE
- Jetson Orin Nano Super 8 GB.
- 2x Arducam AR0234 (global shutter, USB3) = stereo depth pair.
- 1x Arducam IMX477P (12 MP wide, USB3) = scene camera.
- USB mic in, bone-conduction/speaker out. Cameras mounted on a 3D-printed
  glasses frame (see docs/HARDWARE.md).

NON-NEGOTIABLE RULES (carried from the project — do not violate)
- TensorRT engines are device-specific: build them HERE on the Jetson, never
  cross-compile. Colab/PC only produce ONNX.
- The privacy gate is a HARD precondition of every agent call.
- The fast safety loop must stay non-ML and independent of the agent.
- ONNX exports use dynamo=False.
- torch on Jetson must be NVIDIA's Jetson wheel (CUDA), NOT the pip default.
- Speech (wake/STT/TTS) runs on CPU.

WHAT I WANT YOU TO DO
Walk me through bring-up step by step, verifying each stage before moving on.
Use the scripts already in the repo; fix them if the device reveals issues.

  STEP 1. Set performance mode: sudo nvpmodel -m 0 && sudo jetson_clocks.
  STEP 2. Run ./scripts/setup_jetson.sh. Watch for the torch-on-Jetson caveat —
          confirm `python3 -c "import torch; print(torch.cuda.is_available())"`
          is True; if not, help me install the correct NVIDIA Jetson torch wheel.
  STEP 3. Help me get the model artefacts into /opt/argus (yolov8s-worldv2.pt,
          yoloworld_640.onnx, Gemma 4 E2B GGUF + mmproj, Piper voice). Tell me
          exactly which to scp from the PC vs download here.
  STEP 4. Build engines: ./scripts/build_engines.sh. Confirm the YOLO-World
          engine appears in /opt/argus/engines.
  STEP 5. Identify camera indices (v4l2-ctl --list-devices), update
          /opt/argus/config/argus.yaml, then run scripts/calibrate_stereo.py and
          write baseline_m/focal_px back into the config.
  STEP 6. Start Gemma: ./scripts/run_llama_server.sh; verify /health and a test
          image query.
  STEP 7. python -m argus selftest — get every line to PASS (or explain why a
          non-fatal one can stay).
  STEP 8. python -m argus run --no-audio first (validate the fast safety loop by
          walking toward a wall), then the full python -m argus run.

DEBUGGING STYLE
- Diagnose root causes; don't paper over errors. Check logs in /opt/argus/logs.
- When you change a repo file, explain why and keep it consistent with the
  existing module layout (argus/*.py).
- Be economical: this is an 8 GB edge device. Watch memory with tegrastats.

Start by reading the files listed above, then give me a short bring-up plan and
begin at STEP 1.
```

---

### Tips
- Run Claude Code from inside `~/Argus/ARGUS` so it can read the repo directly.
- Keep `tegrastats` open in another pane to watch RAM/GPU while bringing models up.
- If you change config on-device, the live copy is `/opt/argus/config/argus.yaml`
  (the repo's `config/argus.yaml` is just the seed template).

# Jetson Session 2 — Camera Rig Bring-up Prompt

The first Jetson session finished the software stack: JetPack 6.2, Jetson torch
with CUDA, the YOLO-World TensorRT engine, all five model artifacts, a native
llama.cpp build with the vision projector, a healthy Gemma 4 E2B server, and
`argus selftest` passing every fatal check. It ended blocked on two things:
**the cameras were not connected** and **stereo calibration was deferred**.

Both are now unblocked — the three cameras are mounted on the 3D-printed goggle
frame (photos in `Portotype pics/`) and the rig is being connected to the Jetson.

Copy everything in the box below into a **new Claude Code chat running on the
Jetson**, from inside the repo root.

---

```
You are the jetson-agent for ARGUS — AI-powered smart glasses for blind and
visually impaired users — running on an NVIDIA Jetson Orin Nano Super (8 GB,
JetPack 6.2, L4T R36.4.7). A second Claude instance (desktop-agent) works on the
same repo from a desktop and cannot see your terminal. You cannot see its
terminal. docs/AGENT_LOG.md is the ONLY channel between you.

FIRST, IN THIS ORDER, BEFORE DOING ANYTHING:
  1. git pull                       <- MANDATORY. desktop-agent changed the
                                       SAFETY PATH since your last session.
  2. docs/AGENT_LOG.md              <- read the whole log, oldest to newest. The
                                       last entry is addressed to you and ends
                                       with three questions (Q1-Q3) to answer.
  3. docs/KNOWN_GAPS.md             <- what is NOT done; your checklist is §A
  4. docs/HARDWARE.md               <- the rig, and why rigidity matters
  5. docs/CALIBRATION.md
  6. docs/JETSON_DEPLOYMENT.md

WHAT CHANGED WHILE YOU WERE AWAY (re-verify, do not assume)
- Speaker is now non-blocking with NORMAL/WARN/DANGER priorities. The fast loop
  previously froze for the whole duration of every spoken warning; it no longer
  does. DANGER preempts in-flight speech; stale safety phrases are dropped.
- camera.max_skew_ms (default 12) now ENFORCES the stereo skew that was
  previously measured and ignored. Over-limit pairs are dropped before depth.
- The fast loop reports when it runs below 80% of safety.tick_hz.
- New off-device tests: ARGUS_HOME=/tmp/argus_test python3 -m pytest tests/ -q

PROJECT IN ONE PARAGRAPH
Two-Speed architecture. A non-ML FAST loop (stereo depth from the two AR0234s ->
geometric safety reflex -> immediate audio) runs continuously for obstacle and
drop-off warnings. An event-driven SLOW loop handles spoken questions: wake word
-> Whisper STT -> MANDATORY privacy gate (face blur) -> Gemma 4 E2B via native
llama.cpp -> optional find_object(name) -> YOLO-World -> fuse box with depth ->
Piper TTS. Entry point: python3 -m argus.

CAMERA ROLES — do not mix these up
- 2x AR0234 (2.3 MP global shutter), left and right frame edges: THE STEREO PAIR.
  These produce ALL depth. Safety-critical.
- 1x IMX477 (12 MP), centre, above the bridge: THE SCENE CAMERA. This is the
  frame the agent and YOLO-World see. It contributes NOTHING to depth.
- There is no SLAM anywhere in the codebase (KNOWN_GAPS B3). Do not implement it
  this session.

NON-NEGOTIABLE RULES (block any plan that violates one, whoever proposed it)
- Run as the argus user with sudo. Never root.
- TensorRT engines are built HERE on the Jetson, never cross-compiled.
- Gemma runs via the native llama.cpp build with mmproj, not a text-only image.
- torch must be NVIDIA's Jetson wheel (CUDA), never the pip default.
- The fast safety loop stays non-ML and independent of the agent.
- The privacy gate is a hard precondition of every agent call. privacy.require_gate
  stays true.
- /etc/fstab edits use nofail on optional mounts.
- ML benchmarking requires nvpmodel -m 0 + jetson_clocks CONFIRMED (your last
  session found the device in MAXN_SUPER mode 2, not mode 0 — check again).

WHAT I WANT YOU TO DO THIS SESSION

  STEP 1. Confirm max performance mode: sudo nvpmodel -m 0 && sudo jetson_clocks.
          Report what mode it was in beforehand.

  STEP 2. CAMERA DISCOVERY (answers Q1 in the log).
          v4l2-ctl --list-devices
          Confirm the reported names contain the discovery hints "AR0234" and
          "IMX477" (argus.yaml: camera.stereo_name_hint / wide_name_hint). If
          they report different strings, discovery silently falls back to
          resolution grouping and can mis-assign the wide camera — fix the hints
          rather than pinning indices. Then:
          python3 -m argus selftest
          and confirm it now finds three cameras. Verify all three open at once
          without frame drops (USB bandwidth: they are all USB 3.0 on one host).

  STEP 3. STEREO SKEW (answers Q2 in the log).
          Report the observed distribution of skew_ms and how many pairs the new
          12 ms limit drops. If it drops a large fraction, do NOT just raise the
          limit — tell me the numbers first and we decide together. This
          threshold is the difference between honest depth and confident garbage.

  STEP 4. CALIBRATION — but check the mounts first.
          STOP and confirm with me that the ball-joint camera mounts have been
          physically locked down. The calibrator handles any FIXED geometry
          (toe-out, wide baseline, non-coplanar, swapped ports) but cannot handle
          geometry that moves afterwards, and a knocked camera produces depth
          that is wrong without looking wrong. Do not calibrate a rig that can
          still shift.
          Then:
          python3 scripts/calibrate_stereo.py --square-mm <measured> [--headless]
          Iterate until RMS < ~0.6 px and vertical alignment error < ~1 px.
          Then: python3 scripts/calibrate_stereo.py --verify
          against a tape measure at 0.5 m, 1 m, 2 m, 3 m. Report the recovered
          baseline_m and focal_px, and the measured-vs-true error at each
          distance. Everything depth-related is untrustworthy until this passes.

  STEP 5. FAST LOOP ON REAL HARDWARE.
          python3 -m argus run --no-audio
          Walk the rig toward a wall. Expect WARN at 1.5 m and DANGER at 0.7 m.
          Report the achieved fast-loop Hz from the new monitor. Tune
          depth.fast_downscale if it is below 10 Hz. Then tune the safety
          thresholds against reality (KNOWN_GAPS B8) — the current values are
          defaults, not field-tested.

  STEP 6. AUDIO + PREEMPTION CHECK.
          python3 -m sounddevice; set speech.input_device/output_device.
          Then verify the new priority behaviour on real hardware: trigger a long
          agent answer, and while it is speaking, put an obstacle inside the
          danger distance. The answer MUST be cut off and the DANGER warning
          spoken immediately. This is the REQ-NF01 safety guarantee — if it does
          not preempt, that is a blocker, report it.

  STEP 7. SLOW-LOOP LATENCY (answers Q3 in the log).
          Time wake-word -> spoken answer for an image question, broken down into
          STT / privacy gate / Gemma prompt-processing / generation / TTS.
          Gemma currently runs fully on CPU (--device none, -ngl 0) because GPU
          offload exhausted CUDA memory. If total latency is tens of seconds the
          demo is unusable: try a smaller agent.image_max_side (currently 1024),
          a smaller ctx_size, or offloading only the vision projector. Report
          numbers before changing the design.

  STEP 8. MEMORY BUDGET under tegrastats, with everything resident (Gemma server
          + YOLO-World + depth + speech). Report steady-state AND peak. 8 GB is a
          hard ceiling, ~5 GB is the target. Note that a TensorRT engine build is
          far hungrier than running the engine.

REPORTING PROTOCOL — this is how desktop-agent sees your work
Append to docs/AGENT_LOG.md (never edit or delete history) using:

## [YYYY-MM-DD HH:MM] <short title>
**Agent:** jetson-agent
**Status:** in-progress | blocked | done
**What I did:**
**Result / verification:**
**Stuck on / needs input:** (only if blocked)
**Next step (proposed):**

Reference the entry you are replying to explicitly. Commit and push after each
meaningful step so desktop-agent can review — it cannot see anything you do not
push. Post measured numbers, not impressions: desktop-agent has no access to
this hardware and will only trust what is in the log or in the repo.

DEBUGGING STYLE
- Diagnose root causes; don't paper over errors.
- Update docs/KNOWN_GAPS.md when you close or discover a gap.
- Be economical: 8 GB unified memory. Keep tegrastats in another pane.
- If a step needs an interactive sudo password, stop and ask me to run it.

Start by pulling, reading docs/AGENT_LOG.md in full, then give me a short plan
and begin at STEP 1.
```

---

### Notes for the human running this

- The live config is `/opt/argus/config/argus.yaml`; the repo's
  `config/argus.yaml` is only the seed template.
- Keep `tegrastats` open in another pane throughout.
- **Lock the camera mounts before STEP 4.** Re-run calibration after any
  re-seat, knock, or frame flex — the rectification is only valid for the
  geometry it was measured on.
- STEP 4 and STEP 6 both need a person physically moving the rig; the agent
  cannot do them alone.

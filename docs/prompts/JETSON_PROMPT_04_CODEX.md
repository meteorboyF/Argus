# JETSON PROMPT 04 (Codex): take ARGUS from "works in the lab" to a dependable prototype

You are a coding agent running **on the ARGUS Jetson Orin Nano Super 8 GB**:
user `argus`, repo `~/Argus/Argus`, device assets in `/opt/argus`. You have a
shell, the cameras, the GPU, the monitor (`DISPLAY=:0`) and a USB headset.

A human, **Fardeen**, is in the room and does everything physical: cables,
holding a book or chessboard, wearing the rig, walking. ARGUS is a wearable
assistant for blind people. People will walk with it, so correctness and
honesty matter more than speed.

Previous sessions were run by Claude Code. You continue from exactly where
they stopped, and you leave things so Claude, or another agent, can continue
from where *you* stop.

---

## 0. Start-up (always, in this order)

1. Read `AGENT_HANDOFF.md` (the continuation plan, gotchas, open work),
   `STATUS.md`, `docs/FINDINGS_2026-10.md` (device findings with evidence),
   `ARCHITECTURE.md`, `progress.md`, and the last ~15 entries of
   `DECISION_LOG.md`.
2. Run `git log --oneline -20`, `git status` and
   `python3 -m pytest -q tests/` (expect 295+ passing, ~22 s). Preserve any
   uncommitted user work.
3. Run `python3 -m argus doctor --quiet`. If cameras moved, re-bind the roles
   (section 3, rule 9).
4. Write a short plan: the tasks from section 4 you will attempt, each tagged
   **[AGENT]** (you alone) or **[HUMAN]** (needs Fardeen). Start the first
   **[AGENT]** task without waiting for approval.

---

## 1. What exists

Read the files for detail; this is the map.

- **Fast safety loop** (`argus/orchestrator.py::_fast_loop`):
  - stereo 60 fps, skew gate;
  - CUDA SAD depth (`depth.backend: cuda_sad`) or VPI SGM (`vpi_sgm`);
  - `argus/hazards.py`: corridor, ground plane, body/head bands, drop-off,
    head-motion gate, alarms;
  - `argus/movers.py`: YOLO11n TensorRT, tracker, range Kalman, TTC;
  - `argus/safety.py`: legacy approach zones.
  - **All metric hazard speech is disabled until
    `/opt/argus/config/stereo_calib.npz` exists. It does not exist yet.**
- **Slow loop:**
  - `argus/wake_stt.py`: wake on "Argus" via VAD + Whisper;
  - `argus/dialogue.py`: rule router + Gemma JSON router + skills;
  - `argus/reading.py` + `argus/ocr.py`: reading;
  - `argus/seat.py`: seat guidance;
  - grounding;
  - privacy-gated scene description.
- **Speech:** `argus/arbiter.py` (P0 DANGER … P4 SYSTEM) over Piper;
  `argus/vad.py`; `argus/earcons.py`.
- **Tools:**
  - `argus doctor [--snap|--skew-test|--bandwidth|--focus]`,
  - `argus align`,
  - `scripts/calibrate_stereo.py --coach/--verify`,
  - benches: `scripts/bench_vlm.py`, `scripts/bench_depth.py`,
    `scripts/mem_budget.py`.
- **Demo/study:**
  - `python3 -m argus run --demo --record out.mp4`;
  - a JSONL session log in `reports/sessions/`;
  - `scripts/summarise_session.py`.

---

## 2. Non-negotiable rules

1. **Fail closed.**
   - No CPU depth fallback or torch/Ultralytics grounding in the runtime.
   - No metric or danger speech without a calibration that passed the gate.
   - No object distance without a verified projection.
   - Invalid or stale depth is UNKNOWN, never CLEAR.
2. **Privacy.**
   - The face gate runs before any image goes to Gemma.
   - Sensitive text (cards, BD mobiles, OTP/PIN) is masked before logging,
     display or any model call.
   - Demo face blur is display-only and never replaces the gate.
3. **Safety decisions are rules over geometry,** never a language model. The
   slow loop may configure the fast loop (the seat whitelist) but can never
   disable it. Hazards preempt all speech.
4. **Platform.**
   - Stay on L4T R36.5.2 / JetPack 6. No reflash; no JetPack 7, ROS 2 or
     Isaac. No new hardware.
   - `sudo` only for a specific system operation, and ask Fardeen to type it.
     Never store, request or bypass a password.
5. **Evidence, not assertion.** Every "works" claim names:
   - the backend actually loaded,
   - median and p95 latency,
   - peak RAM (RSS and `MemAvailable`),
   - GR3D for GPU work,
   - the over-current throttle delta (`argus.doctor.oc_events()`).

   Save a JSON report under `reports/`.
6. **Small verified steps.** For each feature, in order:
   1. unit tests with synthetic data (see the existing ray-cast, fake-speaker
      and fake-orchestrator patterns in `tests/`),
   2. a live device check,
   3. updated docs (section 5),
   4. a conventional commit ending with your attribution line,
   5. `git push origin main`.
7. **Keep `orchestrator.py` thin.** New behaviour goes into a module with a
   clear interface; the orchestrator only wires.

---

## 3. Lessons already paid for (do not repeat them)

1. **Look at the evidence yourself.**
   - Save frames to `/tmp/argus_snap/` (run `mkdir -p` first: `/tmp` is
     wiped at boot, and `cv2.imwrite` fails silently; assert its return).
   - If you can view images, view them. If not, compute metrics (brightness,
     Laplacian sharpness, match counts) and describe what they imply.
   - Never trust a number from a frame you have not sanity-checked. A
     "perfect" 0 px alignment came from a camera filming its own monitor,
     and another from a rig being handled.
2. **Human steps are guided tools, not silent windows.** Every script a
   person interacts with uses `argus.guide.Guide`:
   - it speaks through Piper (the person cannot see the monitor) and shows
     the same words large on screen;
   - it gives continuous feedback;
   - it ends with a spoken DONE or FAILED plus the reason.
3. **Ask the human once, precisely,** in one message:
   - what to do,
   - the exact command you will run,
   - what they will hear or see,
   - how long it takes,
   - what value you need back.

   Then wait. While waiting, work on an **[AGENT]** task.
4. **Tolerances must exceed measurement noise and what a hand can hold.**
   Otherwise a guidance loop never ends. Hysteresis while the person locks
   or holds still.
5. **pycuda contexts are per thread.** Always use `argus.trt_runner.TRTRunner`,
   which pushes the primary context per call.
6. **Never `pkill -f` a pattern that appears in your own command line;** it
   killed the agent's own shell twice. Stop ARGUS with
   `kill -INT $(pgrep -f "^python3 -m argus run")`, which saves the
   recording and log.
7. **Measurement tools buffer results, not frames.** Buffering raw frames
   OOM-killed the board.
8. **Default config is `/opt/argus/config/argus.yaml`.** Keep it identical to
   `config/argus.yaml`; copy after each edit.
9. **Camera roles drift when cables move.** Re-bind by hub port. Decide
   left/right by parallax: near objects shift further right in the LEFT
   image.
10. **PulseAudio can revert to the analog jack after reboot.**
    `speech.prefer_usb_audio()` runs at start. Verify with
    `pactl info | grep Default`.
11. **The wide camera (B0459) is fixed-focus with no exposure control.**
    Reading quality depends on the lens focus (`doctor --focus`) and on the
    light. Coach the person; don't blame the model.
12. **Gemma 4 E2B** misreads text in images and ignores tool instructions.
    It routes and phrases; code decides. It once claimed an action it
    couldn't do. The persona forbids that; keep it that way.
13. **The demo window closes only on Esc twice or "Argus, shut down".** Keep
    it impossible to end a wearer's session by one stray key.
14. **llama-server benchmarks send `cache_prompt: false`.**

---

## 4. Tasks, in priority order

Each task has a Definition of Done (DoD). Do not mark a task done until the
DoD is met on the device and documented.

### T1. Fast-loop starvation under the full stack [AGENT, safety-critical]

**Problem.** With Gemma + Whisper + the demo view (face blur, recording)
running, the depth heartbeat gaps were 0.26–0.73 s several times a minute
(`depth_gap` events in `reports/sessions/2026-10-21-124310.jsonl`), and stereo
skew drops rose to 12–20 ms.

**Do:**
1. Instrument per-stage timing in the fast loop (capture wait, rectify,
   disparity, hazards, movers, publish).
2. Measure with each heavy component toggled: Gemma request in flight,
   Whisper decoding, face blur thread, recorder, OCR.
3. Fix the cause. Candidates:
   - a dedicated process for the fast loop (shared-memory frames) or OS
     priority (`os.nice`/`sched_setscheduler` where permitted);
   - capping `onnxruntime` and `ctranslate2` threads (`intra_op_num_threads`,
     `cpu_threads`);
   - lowering face-blur and recording rates;
   - moving numpy-heavy render work out of the GIL-bound path.
4. The depth watchdog (arbiter) must keep reporting gaps honestly.

**Corner cases:**
- a camera drops off USB mid-run (the reader must reopen, the watchdog fire,
  then recover);
- the wide camera is reopened for a 12 MP still while the fast loop runs;
- Gemma hits its timeout;
- OCR on a busy page.

**DoD:**
- A 10-minute full-stack soak with continuous voice queries, reading
  attempts and recording.
- Zero `depth_gap` > 250 ms.
- Fast-loop p95 tick ≤ 100 ms.
- No swap thrash; resident stack ≤ 5.5 GB.
- Report: `reports/soak-<date>.json`.

### T2. Reading acceptance [HUMAN + AGENT]

1. Run `python3 -m argus doctor --focus` with Fardeen holding a page at
   arm's length: it guides him to refocus the B0459 lens.
2. Voice tests:
   - "Argus, what's the name of this book?" (*Crime and Punishment* is
     available),
   - "Argus, read this",
   - "next page", "what is this page about?", "stop", "repeat".
3. Add an exposure strategy for the 12 MP still if the coach shows dark
   pages in normal light. Options: average several frames; choose the
   sharpest of N stills; a brighter capture mode if V4L2 exposes one.
   Measure each option.

**Corner cases:**
- backlit page,
- glare on glossy covers,
- a page held upside down (cls model / 180° retry),
- two-column text (reading order),
- a page partly cut off,
- a hand covering text,
- no text at all,
- very long pages (playback controls must still work),
- sensitive numbers on a bill (spoken to the user, masked everywhere else).

**DoD:**
- A printed English page with CER < 3%.
- < 4 s from capture to first spoken word.
- The book title and author read correctly with good light.
- Each failure mode produces the right spoken guidance.
- Report: `reports/reading-acceptance-<date>.json` with CER per sample.

### T3. Stereo calibration [HUMAN + AGENT]

1. Ask Fardeen, in one message, for:
   - the chessboard square size (measure across 4 squares and divide by 4,
     in mm),
   - the tape-measured lens-to-lens baseline (mm),
   - confirmation that the board lies flat.
2. Run `python3 -m argus align --scene-m <distance to a scene ≥ 3 m>`.
3. Then `scripts/calibrate_stereo.py --coach --square-mm N --baseline-mm B`.
4. Then `--verify` at 0.5 / 1 / 2 / 3 m.

**Rules:**
- Never weaken the gate in `scripts/calibrate_stereo.py::quality_gate`.
- On failure, say which metric failed and recapture.

**DoD:**
- `/opt/argus/config/stereo_calib.npz` saved through the gate.
- Tape error ≤ 5% at ≤ 2 m and ≤ 8% at 3 m (`reports/verify-<date>.json`).
- Calibration health is green on the dashboard.

**Then:**
- choose the depth backend by tape accuracy (SAD vs `vpi_sgm`; update
  `config/argus.yaml`);
- confirm that hazard speech, approach TTC speech, seat steps and VO switch
  on.

### T4. Hazard test walk [HUMAN + AGENT] (needs T1 and T3)

**Setup:** soft obstacles, a head-height box (cane blind spot), a door, a
chair, a step or kerb if safe, and a person walking toward Fardeen. Record
with `--demo --record`.

**Annotate:** write ground-truth intervals (JSON) from the video.

**Score:** `scripts/summarise_session.py --truth`.

**Corner cases:**
- fast head turns (the motion gate must block *new* alarms only);
- textureless walls (depth UNKNOWN → no CLEAR);
- reflective glass;
- low light;
- skew drops;
- an approaching person while ARGUS is mid-answer (the P0 cut must be
  < 50 ms in real audio);
- the seat whitelist must not hide a real obstacle.

**DoD:**
- Zero missed head-level boxes in 10 passes.
- ≤ 1 false alarm per minute walking.
- Warning lead ≥ 2.5 m for static obstacles.
- TTC ≥ 2 s for an approaching person.
- Report: `reports/walk-<date>.json`.

### T5. Seat guidance [HUMAN + AGENT]

10 trials across 3 room layouts by voice ("Argus, find me a seat"). Log
success and time.

**Corner cases:**
- no seat in view (search turning),
- two seats,
- an occupied chair (a person on it: never guide onto a person),
- a seat behind the user,
- lost target.

**DoD:** 8/10 successes; `reports/seat-<date>.json`.

### T6. Wake word quality [HUMAN + AGENT]

1. Measure real-voice recall (50 "Argus…" attempts) and false wakes over 2
   hours of normal room sound, including Bangla and English conversation
   and TV.
2. Tune `argus/wake_stt.py` (name variants, minimum phrase length,
   confidence).
3. Optionally train the openWakeWord "Hey Argus" model off-device
   (`training/hey_argus/README.md`) and compare CPU and accuracy.

**DoD:** ≥ 80% recall and ≤ 0.5 false wakes per hour, documented.

### T7. VLM bake-off and memory soak [AGENT, with Fardeen for frames]

- Capture ~30 frames (10 scenes, 10 signs/labels, 10 pages including Bangla)
  under `reports/bakeoff/frames/`, privacy-gated.
- Compare Gemma 4 E2B QAT against Qwen3.5-2B (Q4_K_M + mmproj) at 280 and
  560 tokens with `scripts/bench_vlm.py`.
- Switch only if the winner is clear and fits memory; log it in
  `DECISION_LOG.md`.

### T8. Bangla [HUMAN for sudo + AGENT]

- Evaluate Tesseract `ben` (ask Fardeen to run
  `sudo apt install tesseract-ocr-ben`) and PaddleOCR-VL-1.6-Bangla GGUF as a
  reading mode that pauses the detector.
- Record CER on the Bangla bake-off pages.
- Bangla STT is out of scope unless measured to work.

### T9. Demo deliverable (M8 DoD)

A 2-minute recording covering:
- the startup summary,
- find-object,
- reading a page,
- an approach warning interrupting speech.

Store the MP4 locally (git-ignored) and commit the session summary.

### T10. Headless service [AGENT]

- A `systemd --user` unit (or a documented script) that starts llama-server
  and ARGUS at boot.
- It restarts on crash (fail closed), logs to `/opt/argus/logs`, and speaks
  "ARGUS ready" or the failure reason.
- No sudo unless Fardeen runs it.

---

## 5. Documentation duty (every feature, every session)

Another agent, including Claude, must be able to continue from your last
commit with no other context. After each completed feature:

1. **`STATUS.md`:** update that component's row (status + evidence path), the
   pins (SHA-256 for any new model or engine) and the test count.
2. **`DECISION_LOG.md`:** append an entry in the existing format:

   ```
   ## <date> <milestone> — <title>
   **Hurdle / problem.** … **Impact.** … **Options considered.** …
   **Resolution.** … **Lesson / consequence.** …
   ```

   Every real hurdle is HCI evidence for the study.
3. **`docs/FINDINGS_2026-10.md`:** add any new *device* finding (hardware
   behaviour, measured numbers, failure causes) with its evidence path. Start
   `FINDINGS_<yyyy-mm>.md` in a new month.
4. **`progress.md`:** what was done, live results, what is blocking. Keep it
   short.
5. **`AGENT_HANDOFF.md`:** keep sections 2–6 true. At the end of every
   session, append a dated **Session log** block:
   - what you changed (commits),
   - what you measured (report paths),
   - what is in progress and how to resume it,
   - the exact next [HUMAN] ask.
6. **Tests:** every new rule or module gets synthetic tests. Never delete a
   test to make the suite pass; fix the cause or document a justified change
   in `DECISION_LOG.md`.
7. **Commits:** conventional messages explaining *why*. Push to `origin main`
   after each verified feature. Never commit MP4 recordings (git-ignored),
   unblurred faces, or sensitive text.

## 6. Talking to Fardeen

- He is often wearing the headset and cannot see the screen. Anything he
  must do mid-task goes through `Guide` (spoken).
- In chat, be concrete:
  - what to do,
  - the command you will run,
  - what he will hear,
  - how long it takes,
  - what you need back.
- Report results with numbers, and say plainly when something failed and
  why.
- If he reports something ("it isn't hearing me", "the screen went dark"),
  check the logs and the kernel (`journalctl -k`) before guessing.

## 7. When the prototype counts as done

- **T1–T5 meet their DoD**, which means:
  - a calibrated rig,
  - a stable fast loop,
  - an accepted test walk,
  - accepted reading,
  - accepted seat guidance.
- **T6 measured.**
- **T9 recorded.**
- **The documentation in section 5 is current.**

Until then, `STATUS.md` says "partial" wherever the DoD isn't met on the
device.

Start now with section 0, then T1.

# ARGUS progress

## Power-off checkpoint — 2026-10-29

Saved restart instructions and latest state in AGENT_HANDOFF.md. Operational
recording UI completed; conversation prompt and conceptual book/title routing
improved (84tests). Latest OCR motion uses120ms pairs; live capture succeeded:
two ready checks,5lines OCR in6.747s (45reading tests). Text accuracy/page Q&A
and spoken conversational flow remain to test. No recording active.

## Feature recordings — 2026-10-29

Reading take01 saved146.193s but failed coaching, no reading acceptance.
Fixed premature "capturing" cue, sensor-noise/background-sensitive motion,
and approved-frame vs captured-frame mismatch for stereo OCR.418 tests pass.
Take02 prepared; keep failed take as evidence. Conversation/privacy clips
still next, metric warnings/wideSLAM remain blocked/unimplemented.

Seat take01 finalized:214.274s,video/audio verified. Direction outputs plus
target loss observed. Reading/page-agent take01 started separately with left
POV and local OCR preview. No finalized reading clip yet.

User requested individual clips. Plan docs/FEATURE_RECORDING.md; first seat
direction take launched using new --demo-source left.29 relevant tests pass.
No finalized clip/success yet. Next OCR,conversation,privacy. Metric incoming
warnings/navigation blocked by rejected calibration; wide SLAM unimplemented.
Recorder omits human mic audio; user informed. Recording directory:
reports/feature-recordings/2026-10-29/.

## 2026-10-29: calibration code diagnosis

User authorized trying alternative detector. Added explicit sector calibration
option, recorded in observation/candidate metadata, no classic subpixel pass
on SB corners.46 calibration tests pass; gates unchanged. Fresh sector trial
next with23mm squares and saved observations; not accepted until checks pass.

Stationary burst checked visually/numerically:grainy images,visible centre
seam,~7px global motion. Sector detector gives lower motion-compensated local
repeatability errors:left.368 vs classic.570px,right.216 vs.311px. This is
repeatability,not absolute corner accuracy or accepted calibration. No detector
switch/activation yet; report reports/stationary-board-2026-10-29.json.

Stationary stereo burst test launched for user (SPACE starts).16 pairs, raw
images private under/tmp, numeric report `reports/stationary-board-2026-10-29.json`.
Classic/sector and local repeatability comparison; synthetic smoke passes.
No live result/activation yet; preview owns cameras until Q/Esc.

Robust live25-view session rejected, but starts agree and both tilt directions
pass. Offline check:board-related residuals correlate.907 across cameras;
lens changes barely help. View20 is worst, but excluding it diagnostically
still leaves~1.22px mono/1.34px stereo. Learned target geometry improves some
held-out views but not view20, suggesting inconsistent target/localization
errors, not just one bad frame or a stable warp correction. No raw images
to distinguish flex/blur. Keep mounts fixed; next well-lit stationary corner
repeatability check or verified flat target. No calibration activation.

Implemented optimized multi-start intrinsics and cross-start stability check;
added explicit top/bottom then side-to-side perspective guidance and completion
requirement. No RMS/held-out/baseline limit loosened.413 tests pass. Saved
capture replay now fits99.396mm but still rejects accuracy/held-out/tilt checks;
`reports/calibration-replay-2026-10-29-robust.json`. No active calibration.

Latest23mm session failed severely. Offline analysis reproduces failure but
demonstrates numeric initialization instability:600/800/1000px starting guesses
all optimize to99.396mm baseline and1.089px stereo, vs default2668mm/5.374px.
This is not accepted (mono.677/.636 also above gates). No production fitter
change/activation yet. Pair skew remains low and joint correspondence replay
unchanged. New reports: calibration-replay, calibration-target-analysis and
calibration-initialization, suffix2026-10-29-23mm-live. Next robust initialization
and better out-of-plane pose guidance; no evidence to blame a specific mount.

Fixed-code live capture improved baseline to 98.4 mm but still failed RMS
and held-out gates. Saved observations enable offline study. Lens-model
variants barely help; train-only released-board geometry cuts held-out errors
roughly in half and transfers across cameras. Stronger evidence for target
geometry/localization mismatch than a particular camera mount; not physical
proof of warp. `reports/calibration-target-analysis-2026-10-29.json`.
Next check board flatness/hinge and horizontal/vertical square spacing. No
calibration activated or gates relaxed.

Several live attempts rejected, including post-mount adjustment (15 views,
full coverage, impossible 83.8 cm fitted baseline). Reproduced independent
7x7-grid corner numbering mismatch across diagonal poses. Added joint stereo
ordering with ambiguity rejection and fresh history pairing requiring both
sources to advance. Numeric observations now checkpoint every accepted view;
offline diagnostic replay added. No quality gates relaxed or active calibration
written. 408 tests pass; synthetic 101 mm rig recovered 100.984 mm and passed
held-out validation. Live confirmation pending; old candidates lack observations
to establish which defect caused them. Evidence:
`reports/calibration-code-check-2026-10-29.json`.

## 2026-10-28 continuation

Calibration attempt (7x7 corners, 24 mm squares, ~101 mm baseline) ended at
2/15. Fixed misleading hold-still coach: removed silent same-cell capture
veto while retaining pose diversity/final coverage gates; shared readiness,
clear cooldown/duplicate prompts, SPACE capture. Both-camera steadiness is
checked. Completion requires 70% grid coverage, matching final gate. No
accepted calibration; restarting guided capture for user.

User accepted current reading for now and requested moving on. Reliability
profile (20 s per phase) finished: baseline clean, OCR 316 ms gap, combined
389 ms gap / tick p95 103.9 ms. No operation errors. Prior long outages did
not reproduce; old logs identify missing pairs during those intervals, not
blocked fast-loop work. T1 still partial. Report:
`reports/reliability-stereo-ocr-2026-10-28-summary.json`. Calibration board
grid, square size and lens baseline requested; cameras now released.

Added `run --demo --ocr-preview` for the user's ongoing reading test: local
live source/captured source/masked OCR result window, same voice-triggered
recognition and camera readers, no extra OCR load. It is separate from the
demo recorder. Reading accuracy remains partial.

User preferred stereo OCR in three live comparisons. Stereo returned more
characters and was faster in all three; retain left source. Raw CER favored
wide but comparison used title-only reference against different full-frame
text, so accuracy ranking is unresolved. Need matched crops/verified text;
no 12 MP comparison yet. `reports/ocr-camera-comparison-2026-10-28.json`.

All cameras passed doctor after a left-cable reseat. Human 600.2 s soak failed
with eight depth outages, three ~20.46 s; model healthy, no page-read success.
See `reports/soak-human-2026-10-28-continued-summary.json`.

User requested stereo OCR / wide SLAM with focus untouched. Implemented OCR
from the left live reader for both coaching and capture (960x600), preserving
camera modes and stereo pairing. Both YAML copies updated. 393 tests pass;
live frame/pair-continuity smoke passed; actual page accuracy pending.
Wide metric tracking is NOT implemented: stereo and cross-camera calibration
are needed first. Next: seated page trial, investigate late camera outages,
then calibration/tracker work. Current stereo VO and walking gates unchanged.

Updated: 2026-10-21. Short and honest. `STATUS.md` holds the per-component
table, `AGENT_HANDOFF.md` the continuation plan, and
`docs/FINDINGS_2026-10.md` the device findings with evidence.

## Done in the October sessions (36 commits, 9bd2475..5a3964f)

- **Rig tools:**
  - `argus doctor` (cameras, USB3, fps, skew, RAM, OC throttling, llama-server,
    Piper; `--snap`, `--skew-test`, `--bandwidth`, `--focus`);
  - spoken stereo alignment (`argus align`);
  - spoken calibration coach with a held-out quality gate.
- **Memory/VLM:**
  - QAT Gemma + vision-only projector on GPU, token-budget images (describe
    ~2.2 s);
  - grounding without torch (−1.3 GB).
- **Fast loop:** corridor hazards (head-level, drop-off, motion gate, alarm
  hygiene), YOLO11n TensorRT approach TTC, VPI SGM backend option. All silent
  until calibration.
- **Speech:** P0–P4 arbiter, Silero endpointing, earcons, USB headset
  auto-select, wake on "Argus" without a trained model.
- **Conversation:** rules plus Gemma routing, context, follow-ups, persona,
  shut down by voice.
- **Reading:** OCR, 12 MP still, document locate + contrast crop, coach that
  guides on light/glare/blur/position, diagnose and retry, masked page
  questions.
- **Seat guidance**, closed loop with a corridor whitelist.
- **Demo:** first-person view, MP4 recording with audio, JSONL study logs,
  `summarise_session.py`.
- 321 tests pass after the Codex T1 changes.

## Live results with the user (2026-10-21)

- **Voice conversation works:**
  - wake on "Argus", then "Yes?", then a question;
  - "what is in front of me" answered in 2.4–3.0 s, chat in 1.6 s;
  - follow-ups without the wake word.
- **Reading a real book cover failed.** The wide camera's lens is fixed-focus,
  set far, and a window behind the book made it dark. Code now coaches for
  this; the lens refocus is pending.

## What is blocking

1. **Fast-loop stalls under the full stack** (0.26–0.73 s depth gaps).
   Safety-critical; next engineering task.
2. **No stereo calibration** (needs the user: chessboard square size,
   lens-to-lens baseline, flat board). Blocks every metric safety feature.
3. **Wide-camera focus** for reading (`doctor --focus` with the user).
4. Not yet tested with the user: reading by voice, seat by voice, walking
   hazards, wake-word false accepts per hour.

## Codex continuation, 2026-10-21

- Added per-stage fast-loop timings and completed/open-outage accounting.
- Fresh bounded stereo pairing avoids the latest-frame phase mismatch; both
  source sequences must advance. Stale source depth and cached depth expire.
- CPU inference pools are bounded; display colourisation/sysfs reads are off
  the safety thread; llama batch/microbatch reduced from 1024 to 256 after two
  CUDA allocation failures.
- Short combined synthetic load: 34.5/56.7 ms median/p95 tick, 3856 MiB RSS,
  292 MiB minimum MemAvailable, GR3D peak 99%, OC3 +1654. Separate busy OCR
  still caused one >250 ms outage. Reports: `t1-profile-final-2026-10-21.json`
  and `t1-failures-2026-10-21.json` under `reports/`.
- T1 remains partial. Fardeen is unavailable for the human soak; guided
  `scripts/soak_fast_loop.py` is prepared. Automated offscreen stress completed; an
  8 GiB NVMe swap trial awaits activation. Focus/calibration/walking pending.

- Automated 600.6 s pre-swap stress: 42.8/79.8 ms median/p95 tick, two depth
  outages (519/262 ms), 4196 MiB RSS peak, 401 MiB minimum available RAM,
  GR3D 100%, OC3 +51587. Server survived; T1 failed. Summary:
  `reports/soak-2026-10-21-summary.json`.
- Prepared and verified `/opt/argus/swapfile.nvme-8g` (8 GiB, mode 0600,
  swap signature, no holes). Waiting for Fardeen to activate at priority 0;
  exact command and post-swap comparison in `docs/operations/MEMORY_TRIAL.md`.

## NVMe follow-up, 2026-10-21

- Fardeen activated 8 GiB NVMe swap, temporary priority 0; no boot changes.
  Paired 600 s run: 43.7/80.6 ms median/p95 tick, two outages (302/574 ms),
  4118 MiB peak RSS, 264 MiB minimum available, GR3D 99%, OC3 +50752.
  No observed NVMe occupancy; no timing benefit demonstrated.
- Found CTranslate2's optional torch imports. Production excludes them and
  rejects unsupported object labels without live CLIP. No installed package
  edits. Fresh-process STT/VAD check saved 330.5 MiB peak RSS with identical
  synthetic outputs. 321 tests pass. Final 600 s full-stack validation completed (below).

- Final automated run: no >250 ms outages, max heartbeat 202.7 ms; CUDA SAD,
  43.2/81.3 ms median/p95 tick, 4123 MiB peak RSS, 426 MiB minimum available,
  GR3D 99%, OC3 +51511. Torch absent, no reported request errors. Summary:
  `reports/soak-inference-budget-2026-10-21-summary.json`. Timing/RSS subtargets
  passed once; T1 still partial pending human voice/reading and fault checks.
- NVMe swap remains active temporarily at priority 0; no boot configuration
  change. No further sudo action is required for the current experiment.


## Software build, physical tests deferred — 2026-10-21

- User requested build now/test later. Built seat/TRT ownership and occupancy
  safeguards; reading cancellation, bounded still/chunks, column and orientation
  handling; headless supervisor/unit; acceptance scorers/protocols; gated frame
  collector and configurable VLM comparison; optional Tesseract Bangla evaluator.
- 343 synthetic tests pass. Evidence `reports/build-2026-10-21.json`; runbook
  `docs/ACCEPTANCE.md`. No new live performance results. New service not enabled,
  no models/packages downloaded, config/pins unchanged. All live DoDs partial.
- Resume human testing only when requested. Repeat T1 on this build, then focus,
  calibration, reading, walk, seat, wake and demo. Optional PaddleOCR-VL/wake
  training/model selection still depend on evaluation, not guessed compatibility.


## Companion and local mapping software — 2026-10-21

- Built session conversation/memory/preferences/recap, 20 s follow-ups, bounded
  async tasks/cancellation, timers, detector-backed object inventory/search,
  sparse keyframe relocalisation, expiring occupancy map/A*, named places and
  guarded routes. Text/model privacy boundaries tightened; invalid depth is
  UNKNOWN; stale wide frames rejected. `docs/COMPANION.md` lists commands.
- 386 synthetic tests pass; `reports/companion-build-2026-10-21.json`. Both YAML
  copies match; no new models/engines. Barge-in and walking routes default off.
- Live tests remain deferred. Repeat T1 including new CPU/mapping work, then
  focus/calibration/walk/seat/wake acceptance. No new timing/RAM/GR3D/OC claims,
  no surveyed/global SLAM or outdoor-navigation claim. Physical DoDs partial.

# ARGUS progress

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

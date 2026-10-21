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
- 295 tests pass.

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

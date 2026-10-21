# ARGUS agent handoff

Last updated: 2026-10-21 (Codex T1 continuation; see newest git commit).
Read in this order: this file → `STATUS.md` → `docs/FINDINGS_2026-10.md` →
`ARCHITECTURE.md` → the newest entries at the end of `DECISION_LOG.md`. Material
under `historical/` is evidence, not instructions.

## 1. What ARGUS is right now

A wearable assistant for blind users on a Jetson Orin Nano Super 8 GB, running
as two loops:

- **Fast safety loop**, deterministic, 10 Hz: stereo depth, corridor hazards
  (ground plane, body/head bands, drop-offs), approaching people/vehicles
  (YOLO11n TRT + tracker + TTC), head-motion gate, alarm hygiene. **All metric
  speech is off until a stereo calibration file exists** (there is none yet).
- **Slow conversational loop:** wake on the spoken name ("Argus" / "Hey
  Argus"), Silero VAD endpointing, Whisper tiny STT, a dialogue manager (rules
  plus Gemma 4 E2B routing), and skills:
  - describe the scene (privacy-gated image to Gemma),
  - find an object (YOLO-World TRT),
  - read text (coach, 12 MP still, OCR, sentence playback),
  - answer questions about a page (Gemma, masked text only),
  - seat guidance,
  - small talk,
  - shut down.
- **Speech:** a P0–P4 arbiter over Piper, where hazards cut everything, plus
  earcons and USB headset auto-selection.
- **Demo/study:** a first-person view with captions, hazard banner and
  insets, MP4 recording with ARGUS's audio, a JSONL session log and
  `scripts/summarise_session.py`.

## 2. Run it

```bash
cd ~/Argus/Argus
scripts/run_llama_server.sh &                         # Gemma 4 E2B QAT on :8080 (~10 s)
python3 -m argus doctor                               # rig check (cameras, USB3, fps, skew, RAM, OC, llama, Piper)
DISPLAY=:0 python3 -m argus run --demo --record reports/sessions/demo.mp4
#   wake: "Argus, what's in front of me?" | "Argus" -> "Yes?" -> request
#   close: Esc twice within 2 s, or "Argus, shut down" (saves recording + log)
DISPLAY=:0 python3 -m argus run --dashboard --no-mic --ask-file /tmp/ask.txt   # typed questions
python3 -m pytest -q tests/                            # 321 tests; all must pass
python3 scripts/summarise_session.py reports/sessions/<log>.jsonl
```

Guided human tools (each one speaks its instructions, shows them on screen
and ends with DONE or FAILED):

- `python3 -m argus doctor --snap`: worn-orientation snapshots.
- `python3 -m argus doctor --skew-test`: on-screen timecode, true stereo offset.
- `python3 -m argus doctor --bandwidth`: fps matrix.
- `python3 -m argus doctor --focus`: wide-camera lens refocus for reading.
- `python3 -m argus align --scene-m 3 --baseline-m 0.13`: stereo mount
  alignment. Face a scene at least 3 m away; never the monitor.
- `python3 scripts/calibrate_stereo.py --coach --square-mm N --baseline-mm B`,
  then `--verify`.

## 3. Hard rules (do not break)

1. Fail-closed production. No CPU depth fallback, no Ultralytics/torch
   grounding fallback in the runtime.
2. No metric or danger speech from stereo without a gated calibration. No
   object distance without a verified projection.
3. The privacy gate runs before any image reaches Gemma. Sensitive text is
   masked before it is logged, shown or sent.
4. Safety decisions are geometry and rules, never a language model. The slow
   loop may *configure* the fast loop (the seat whitelist) but never disable it.
5. Stay on L4T R36.5.2 / JetPack 6. No reflash, no JetPack 7, ROS 2 or Isaac.
   No new hardware.
6. `sudo` only for a specific system operation; never store, request or
   bypass the password.
7. Evidence, not assertion: backend, median/p95 latency, peak RAM and GR3D
   (plus the OC throttle delta) in a JSON report under `reports/`.
8. One verified feature at a time. Then update `STATUS.md`, `progress.md`,
   this file, `DECISION_LOG.md` (and `docs/FINDINGS_*.md` for device
   findings), commit with a conventional message, and push to `origin main`.

## 4. Gotchas that already cost time

- **Camera roles drift on recabling.** Re-run `doctor` and check left/right by
  parallax (near objects shift further right in the LEFT image). Current
  binding: left `2-1.1`, right `2-1.2`, wide `2-1.4`.
- **`/opt/argus/config/argus.yaml` is the default config** and must match
  `config/argus.yaml`. Copy after every edit. Unknown keys now print a
  warning.
- **PulseAudio may revert to the analog jack after a reboot.**
  `prefer_usb_audio()` fixes it at start; check with
  `pactl info | grep Default`.
- **pycuda contexts are per thread.** Use `TRTRunner`, which pushes the
  primary context per call. Never call raw pycuda on worker threads.
- **Never `pkill -f` with a pattern that appears in your own command.** Stop
  ARGUS with `kill -INT $(pgrep -f "^python3 -m argus run")`, which saves the
  recording and log.
- **Don't buffer raw frames in measurement tools** (OOM). Don't open a camera
  that ARGUS already has open.
- **Never measure alignment while the camera faces its own monitor** (video
  feedback). Check every measurement by viewing the frame (`Read` the JPEG).
- **The B0459 lens is fixed-focus.** Reading quality depends on the
  `--focus` session and on the light.
- **`/tmp` is wiped at reboot**, including `/tmp/argus_snap`. `cv2.imwrite`
  fails silently, so `mkdir -p` first and assert its return value.
- **The demo window cannot be closed by one Esc** (by design): use Esc twice
  or "Argus, shut down".
- **Benchmarks:** send `cache_prompt: false` to llama-server, and note OC
  throttling.

## 5. Open work, in priority order

1. **Fast-loop starvation under the full stack** (safety-critical).
   `depth_gap` events of 0.26–0.73 s and 12–20 ms skew drops occur while
   Gemma, Whisper, the demo view, face blur and recording run.
   - Codex added scalar `fast_tick` stages and `depth_ready` events, bounded
     fresh stereo pairing, 250 ms source/cached-depth expiry, CPU pool caps,
     async latest-only previews, and llama batch/microbatch 256.
   - Short final combined tick median/p95 34.5/56.7 ms, 3856 MiB peak RSS,
     292 MiB minimum MemAvailable, peak GR3D 99%, OC3 +1654. A separate busy
     OCR phase still failed >250 ms. T1 is not accepted.
   - See `reports/t1-profile-final-2026-10-21.json` and
     `reports/t1-failures-2026-10-21.json`. Earlier profiles include cached
     Gemma, a no-server interrupted baseline, and a real USB outage; read caveats.
   - Reproduce without stealing desktop focus:
     `python3 scripts/profile_fast_loop.py --offscreen --phases combined --seconds 600 --out reports/soak-automated-offscreen-2026-10-21.json`.
   - Human soak (Fardeen unavailable this session):
     `DISPLAY=:0 python3 scripts/soak_fast_loop.py --seconds 600 --out reports/soak-2026-10-21.json`.
     Stay seated, printed page/headset ready; Guide prompts every 45 s,
     display/recording, final DONE/FAILED. Report never auto-claims T1 complete.
   - Fardeen activated the 8 GiB NVMe file at priority 0; zram priority 5 and
     swappiness 60 remain. The paired 600 s run still had two outages (302/574
     ms) and no observed NVMe occupancy at five-second samples. No timing
     benefit demonstrated; no fstab edit. See `reports/nvme-swap-comparison-2026-10-21.json`.
   - Found unnecessary torch imports from CTranslate2 conversion/spec modules.
     `runtime_imports.py` excludes torch in production; preloaded torch requires
     a fresh process. Unknown production object labels cannot use live CLIP.
     Fresh-process STT/VAD test saved 330.5 MiB peak worker RSS with identical
     synthetic outputs. CTranslate2 is pinned to installed 4.8.0. Final combined
     validation passed timing/RSS subtargets once (zero >250 ms gaps); see
     `reports/soak-inference-budget-2026-10-21-summary.json`. Human/fault
     acceptance remains pending.
   - Profile per thread if remaining tails persist.
   - Candidate fixes:
     - give the fast loop its own process or raised priority;
     - move face blur and recording to lower priority or a lower rate;
     - limit Whisper and onnxruntime threads;
     - avoid GIL-heavy work in the render thread.
   - DoD: no `depth_gap` > 250 ms in a 10-minute full-stack soak; record RSS
     and swap for the M3 memory target (≤ 5.5 GB resident).
2. **Lens focus and reading acceptance** (needs the user).
   - Run `doctor --focus` with a page at arm's length.
   - Then a voice test: "Argus, what's the name of this book?", "Argus, read
     this", "next page", "what is this page about?".
   - DoD: printed English page CER < 3%, < 4 s from capture to first word;
     book title read correctly with the user's back to the window.
3. **Stereo calibration** (needs the user and the board).
   - Ask for the square size, baseline and board flatness.
   - Run `align` (scene ≥ 3 m), then `calibrate_stereo.py --coach`, then
     `--verify` at 0.5 / 1 / 2 / 3 m.
   - DoD: per STATUS, the gate passes, tape error ≤ 5% at ≤ 2 m and ≤ 8% at
     3 m.
   - That unlocks metric hazard speech, approach TTC speech, seat steps, VO
     and the depth-backend accuracy choice (SAD vs `vpi_sgm`).
4. **Test walk** (M4 DoD): soft obstacles, head-level box, door, chair,
   person approaching; precision/recall per hazard via
   `summarise_session.py --truth`. DoD: zero missed head-level boxes in 10
   passes, ≤ 1 false alarm per minute walking.
5. **Seat guidance test** (M7 DoD): 8 of 10 successful approaches across 3
   layouts.
6. **Wake word.** Measure real-voice recall and false wakes per hour (target
   ≤ 0.5/h at ≥ 80% recall). Train the openWakeWord "Hey Argus" model off-device
   (`training/hey_argus/`) to cut CPU.
7. **M3 remainder:** a 30-frame Gemma vs Qwen3.5-2B bake-off at 280/560
   tokens (CER for signs, rubric for scenes), and a 10-minute memory soak.
8. **Bangla:** evaluate Tesseract `ben` (needs `sudo apt install
   tesseract-ocr-ben`; ask the user) and PaddleOCR-VL-1.6-Bangla GGUF as a
   mode switch; record CER.
9. **Demo deliverable (M8):** a 2-minute recording covering the startup
   summary, a find-object, reading a page, and an approach warning
   interrupting speech (the last needs calibration).

## 6. Where things live

| Path | What |
|---|---|
| `argus/orchestrator.py` | lifecycle, fast loop, listen loops, wiring (keep thin: add modules, not code here) |
| `argus/dialogue.py` | rule router + Gemma router + skills + conversation context |
| `argus/wake_stt.py`, `argus/vad.py`, `argus/speech.py`, `argus/arbiter.py`, `argus/earcons.py` | voice in/out |
| `argus/reading.py`, `argus/ocr.py`, `argus/sensitive.py` | reading mode, OCR, masking |
| `argus/hazards.py`, `argus/movers.py`, `argus/safety.py` | fast-loop rules (corridor, approach; legacy reflex) |
| `argus/depth.py`, `argus/cuda_stereo.py`, `argus/vpi_stereo.py`, `argus/calib_health.py` | depth |
| `argus/timing.py`, `argus/stereo_pairing.py`, `argus/depth_freshness.py`, `argus/cpu_budget.py`, `argus/depth_preview.py` | T1 timing, source age, pairing and resource bounds |
| `argus/seat.py`, `argus/navigation.py`, `argus/slam.py` | guidance |
| `argus/grounding.py`, `argus/vocab.py`, `argus/trt_runner.py` | detectors |
| `argus/doctor.py`, `argus/align.py`, `argus/guide.py`, `argus/timecode.py` | rig tools; `Guide` = spoken + on-screen instructions |
| `argus/demo_view.py`, `argus/dashboard.py`, `argus/telemetry.py` | views, bus, `SessionLogger` |
| `scripts/` | calibration, benches (`bench_vlm.py`, `bench_depth.py`, `mem_budget.py`), model tools, `summarise_session.py` |
| `reports/` | JSON evidence; `reports/sessions/*.jsonl` study logs (MP4s are git-ignored) |
| `/opt/argus/{models,engines,exports,config}` | device assets; hashes in `STATUS.md` |

## 7. Prompts and research

- `docs/prompts/JETSON_PROMPT_04_CODEX.md`: the current task prompt
  (priorities T1–T10, DoDs, corner cases, documentation duty).
- `docs/RESEARCH_2026-10.md`: the design targets from the October research.

## 8. Session log

### 2026-10-02 → 2026-10-21 (Claude Code, JETSON_PROMPT_03)

- **Commits:** `9bd2475..5a3964f` (36) plus this documentation commit. See
  `git log`.
- **Measured** (all under `reports/`):
  - `doctor-*`, `usb-bandwidth-*`, `worn-orientation-*`, `align-*`;
  - `vlm-m3-*`, `mem-budget-*`, `depth-backends-*`, `reading-latency-*`,
    `dialogue-routing-*`;
  - `sessions/*.jsonl` (voice tests).
- **Live with the user:**
  - voice conversation works (wake on "Argus", "Yes?", describe, chat,
    follow-ups);
  - a real book cover could not be read (fixed-focus lens + backlight). The
    coach now guides for this.
- **In progress:**
  - nothing half-written;
  - the working tree is clean after the documentation commit.
- **Next:**
  - T1 (fast-loop starvation) can start without the user;
  - the first [HUMAN] asks are `doctor --focus` (reading) and the calibration
    inputs: square size, baseline, flat board.

### 2026-10-21 (Codex, JETSON_PROMPT_04 T1)

- **Changes:** `a7b58b6` (`fix(safety): bound fast-loop contention and expire
  stale depth`, pushed to origin/main): stage/heartbeat timings, fresh bounded pairing, old source/cached
  depth rejection, CPU caps, latest-only previews, batch 256, matrix and guided
  soak tools. No calibration/model/engine changes. 314 tests pass.
- **Measured:** `reports/doctor-2026-10-21-1336.json`,
  `reports/t1-profile-{default,budgeted,budgeted-live,final}-2026-10-21.json`,
  `reports/t1-failures-2026-10-21.json`. Failed runs are retained with caveats.
- **In progress:** automated offscreen stress completed (T1 failed);
  NVMe swap trial requested by the user, prepared and awaiting activation. T1–T5 remain partial; no walking permitted by acceptance.
- **Next [HUMAN] ask:** activate the prepared NVMe swap file using the exact
  command below when available; then rerun paired measurements. Later, headset
  plus printed page for the seated guided soak. Fardeen declined that soak this
  session; don't silently treat synthetic speech/OCR as its replacement.

#### Session outcome / exact resume point

- 314 tests pass. Automated 600.6 s stress: CUDA SAD, 42.8/79.8 ms median/p95
  tick, 4196 MiB peak stack RSS, 401 MiB minimum MemAvailable, GR3D 100%, OC3
  +51587; two outages 519/262 ms. Server stayed alive. T1 is **not accepted**.
  Reports: `reports/soak-2026-10-21-summary.json`,
  `reports/soak-automated-offscreen-2026-10-21.json`.
- The first automated GUI run was interrupted after its demo received double
  Esc. Offscreen mode was added to avoid disturbing desktop use; do not count
  `reports/soak-automated-2026-10-21.json` as evidence of a full run.
- New source-age guard also suppresses hazard speech if mover detection blocks
  long enough to make geometry old. It was added during the baseline; repeat
  final code in the post-swap run. Four synthetic freshness tests cover it.
- `/opt/argus/swapfile.nvme-8g` exists: 8589934592 bytes, mode 0600, 82 fully
  allocated extents, swap UUID `bb6ee86b-f34c-4546-9d39-4c4b7341c424`.
  No activation or fstab change by the agent. Preparation evidence:
  `reports/nvme-swap-preparation-2026-10-21.json`.
- **Exact next [HUMAN] ask (already sent):**
  `sudo chown root:root /opt/argus/swapfile.nvme-8g && sudo swapon --priority 0 /opt/argus/swapfile.nvme-8g`
  Reply activated or the error; never send a password. Check `/proc/swaps`
  before assuming activation. Existing zram priority 5, swappiness 60.
- After activation run the identical 600 s offscreen comparison from
  `docs/operations/MEMORY_TRIAL.md`, save `reports/soak-nvme-swap-2026-10-21.json`,
  compare paging/tails/latencies and update this log. If still failing,
  evaluate safety-process isolation and reducing desktop memory pressure.
- llama-server is running with production QAT model/projector on CUDA, batch
  and microbatch 256 (now the script defaults), logging to
  `/tmp/argus-t1-llama-256.log`. No ARGUS wearable runtime remains running
  after the benches. Do not kill the server before the paired swap trial.

### 2026-10-21 (Codex follow-up: NVMe activated, inference imports)

- **Changed:** production import budget (`runtime_imports.py`), explicit
  unsupported-label handling with honest speech, pin of existing CTranslate2
  4.8.0, `scripts/bench_stt_memory.py`, torch-presence resource samples.
  No installed package patches, new models/engines or YAML changes.
- **Measured:** active priority-0 NVMe verified in
  `reports/memory-after-nvme-activation-2026-10-21.json`;
  `reports/soak-nvme-swap-2026-10-21.json`,
  `reports/nvme-swap-observer-2026-10-21.json`,
  `reports/nvme-swap-comparison-2026-10-21.json` (two >250 ms outages; no
  observed disk occupancy, no timing benefit); fresh CPU STT/VAD comparison in
  `reports/stt-import-memory-2026-10-21.json` saved 330.5 MiB peak worker RSS.
- **Tests:** 321 pass; full combined validation with production import budget
  completed and the final result is appended below.
- **Current system:** 8 GiB NVMe swap active temporarily; existing zram still
  preferred; swappiness 60. No `/etc/fstab` changes. Do not request activation
  again. llama-server uses unchanged QAT weights/projector, batch/microbatch 256.
- **Next [HUMAN] ask:** when Fardeen is available, headset and printed page for
  the already-described seated 10-minute guided voice/reading soak:
  `DISPLAY=:0 python3 scripts/soak_fast_loop.py --seconds 600 --out reports/soak-2026-10-21.json`.
  It speaks/displays prompts every 45 s and DONE/FAILED; collect missed speech,
  reading failures and interruptions. Do not repeat the ask until available.
  Focus/calibration inputs follow; no calibrated/walking acceptance yet.

#### Follow-up final outcome / resume

- Final 600.2 s automated combined run: zero >250 ms depth outages, maximum
  heartbeat interval 202.7 ms, CUDA SAD, tick 43.2/81.3 ms median/p95, peak
  stack RSS 4123 MiB, minimum MemAvailable 426 MiB, GR3D 99%, OC3 +51511.
  Torch absent from all samples, no reported request errors. 321 tests pass.
- Reports: `reports/soak-inference-budget-2026-10-21-summary.json` plus full
  `reports/soak-inference-budget-2026-10-21.json`. Automated timing/RSS
  subtargets passed once. Do not mark T1 complete: synthetic workload,
  uncalibrated geometry, human continuous voice/reading and full fault checks
  are incomplete. Re-run once calibration activates the geometry engines.
- Only the standalone worker saved 330.5 MiB RSS; full-stack peak RSS remained
  comparable. Lower system paging may include warm-cache/desktop effects.
  NVMe alone did not demonstrate a timing benefit. Do not attribute the final
  zero-gap result to disk paging.
- Swap remains active temporarily at priority 0, swappiness 60, zram priority 5.
  No further activation ask, package patches or boot configuration changes.
  llama-server is running batch/microbatch 256, unchanged model/engine pins.
  Profiling runtimes and observers have ended; no cameras are left open.
- Next [AGENT]: review/perform remaining timeout and fault checks without
  weakening gates. Next [HUMAN]: the seated guided soak when available, then
  focus and measured calibration inputs. Preserve the explicit not-done state.

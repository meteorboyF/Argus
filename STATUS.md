# ARGUS implementation status

## Power-off checkpoint — 2026-10-29

Latest OCR120ms motion sampling successfully captured live:5lines in6.747s;
accuracy and page questions unassessed.45reading tests passed. Conversation
server restored; direct advice and conceptual text routing improved;84tests
passed. User powering down Jetson; resume via newest AGENT_HANDOFF.md section.

## 2026-10-29 recording interface redesign

Native demo UI redesigned with large camera, feature headings, readable
masked OCR/transcript and activity states. Tab toggles diagnostics;
`--demo-feature` selects the presentation. Actual OCR now appears inside the
recorded composition. Existing privacy and calibration gates remain.
Live unrecorded reading preview launched successfully; all three cameras
discovered. Offline screenshots are illustrative. Renderer-only benchmark:
median20.2ms,p9522.0ms; full-stack performance acceptance not claimed.

## 2026-10-29 feature-recording session

Reading take01 finalized146.193s(video+audio) but coaching timed out; no OCR
success. User reported misleading hold-still/capturing prompts. Fixed prompt
semantics (framed!=captured), denoised page-ROI motion instead of raw full-frame
differences, and use of the approved stereo frame rather than later live frame.
Brightness/blur/framing gates retained, two consecutive ready checks still
required. Added numeric reading_frame_check telemetry; no text/images in it.
418 tests pass including noise/background-motion, exact-frame OCR and cancelled
result regressions. Real improvement pending take02. Other feature clips pending.

Seat take01 saved/stream-verified:214.274s,H264 960x540+AAC. Log shows
direction outputs and intermittent target loss; not acceptance. Separate
reading/page-agent take01 launched; not finalized. See recording manifest.

User switched to separate feature tests/recordings. Plan in
`docs/FEATURE_RECORDING.md`; manifest and takes under
`reports/feature-recordings/2026-10-29/`. First stationary seat-direction take
launched. No finalized/accepted clip yet. Added --demo-source left/wide for
faithful stereo POV recording (default wide); inference camera routing and
privacy/calibration gates unchanged.29 relevant tests pass. Recorder includes
ARGUS audio/request captions, not human microphone audio. Latest sector
calibration rejected, so metric warnings/walking remain gated. Wide SLAM
remains unimplemented; existing mapping is calibrated stereo. Reading,
conversation and privacy demonstrations pending after seat test.

## 2026-10-29 calibration diagnosis and corrections

Added explicit `--detector sector` calibration option using subpixel SB corners
without classic refinement; default classic retained. Auto-board detection and
observation/candidate metadata use selected detector.46 calibration tests pass,
including synthetic board/blank rejection. Quality gates unchanged. Sector
trial uses23mm,7x7 corners,101mm baseline; live acceptance still pending.

Stationary burst complete:16 pairs,skew<=1.513ms. Images visually grainy,
center folding seam visible; hand-held target moved~7px globally (cannot
call perfectly stationary). After projective compensation, median local
repeatability classic/sector: left.570/.368px,right.311/.216px. Sector
detected15/16 left,16/16 right; classic16/16 left,15/16 right. Median
planar-grid RMS classic/sector:left.767/.677,right.652/.632px; this includes
lens/target effects, NOT calibration acceptance accuracy. Sector improves
repeatability on these frames but no runtime detector change yet. Fold
flatness/geometry not established from photos. Private PNGs in
/tmp/argus-stationary-board-48s2_qfy; report reports/stationary-board-2026-10-29.json.
No activation. Preview may still own cameras until Q/Esc.

Prepared/launched stationary-board diagnostic (SPACE-triggered16-pair burst,
classic vs sector detector, motion-compensated repeatability). Private raw
PNGs under/tmp, numeric report under reports. Synthetic identical-frame smoke
passed both detectors; live results pending. No calibration activation.

Latest robust live capture:25 views,7/9 coverage,tilt spans.175/.427,arrival
skew median.56/max3.55ms. Intrinsic starts agree; baseline98.31mm. Still
rejected:mono1.261/1.316,stereo1.411,held-out.570/1.953px. Offline replay
confirms; tilt/initialization/pair-skew checks do not explain residual failure.
View20 is a clear outlier (~2.0/2.7px mono), but diagnostic exclusion still
gives1.217/1.221px mono and1.337px stereo; not a one-view fix. Original
observations unchanged; no acceptance from pruning.
Lens-model variants barely help. Shared board-coordinate residual pattern
correlation.907; released-board model helps several held-out views but view20
stays bad (>2px), so a single fixed target correction is insufficient. This
supports target/localization inconsistencies (possibly flex/blur), not proof
of a particular physical defect or operator mistake. Raw images were not
saved, so blur/flex cannot be distinguished from corners alone. Reports:
`calibration-replay`, `calibration-target-analysis`, `calibration-residual-pattern`,
`calibration-outlier-sensitivity`, suffix`2026-10-29-robust-live` under reports.
No activation. Next diagnostic: stationary well-lit capture/corner repeatability,
or a verified flat rigid printed target, rather than another identical session.

Implemented multi-start intrinsic fitting: three image-size-based focal guesses
are optimized (not fixed); lowest mono reprojection fit selected independently
of tape baseline. Similar-error starts must agree within10% focal spread for
acceptance. Existing RMS/held-out/baseline gates unchanged. Added perspective
cues from opposite-edge length ratios, invariant to image translation, scale
and flat rotation. Coach requests top/bottom then side-to-side board tilts;
auto-completion/final gate also require cue span>=.08 in both directions.
This is a capture-conditioning heuristic, not proof of pose/metric accuracy.
Checkpoints save tilt cues; offline replay includes stability/tilt diagnostics.

413 tests pass, including bad-start/stability regressions and no-false-tilt
tests; compile/diff checks pass. Latest saved capture now fits99.396mm baseline,
mono.677/.636px,stereo1.089px, but held-out48.19/214.69px and limited top/bottom
cue span.0536 still reject. Evidence `reports/calibration-replay-2026-10-29-robust.json`.
No activation; live validation with stronger out-of-plane pose variation pending.

Newest 23mm live session rejected:15 views/8 cells, stereo5.374px,
baseline2668mm. Offline replay confirms pathological default left intrinsics
fx7037/fy4877 versus right734/746. Three diagnostic optimizations initialized
at600/800/1000px all converge to left671/682,right734/746,baseline99.396mm,
stereo1.089px,mono.677/.636px. Guesses optimized, not fixed; no baseline tuning
or activation. This directly demonstrates initialization sensitivity; stable
fit still fails quality gates. Production fitter has NOT yet been changed.
`reports/calibration-initialization-2026-10-29-23mm-live.json`.
Pair skews1.14–3.33ms; stored joint corner order unchanged on replay. Estimated
plane angles using earlier intrinsics suggest limited out-of-plane variation;
those angles are diagnostic estimates, not ground truth. Prior-intrinsics
diagnostic fit gives stereo.936px/baseline93.54mm, also not acceptance.
Next software step: robust intrinsic initialization/stability diagnostics;
then explicit tilt/distance capture guidance and remaining target/localization
checks. Do not ask user to change a specific mount based on the bad fit.

Latest fixed-code live session: 15 views/9 cells, baseline 98.41 mm (tape
101 mm), stereo RMS .988 px, per-camera .818/.835 px, held-out mean/max
.926/2.971 px. Rejected, no activation. Corners now saved in
`reports/calibration-observations-2026-10-29-fixed.npz`; replay matches.

Offline target analysis fits 12 views, holds out views 5/10/15. Free-k3 and
rational lens models barely improve ideal-grid errors. Releasing target
geometry reduces held-out per-camera RMS: left .80–.97 to .39–.44 px,
right .73–.87 to .30–.39 px. Left-derived target shape transfers to right
held-out views (.32–.40 px), and vice versa (.36–.44 px). Supports systematic
target-related geometry/localization error; does not physically measure warp
or prove its sole cause. Learned millimetre deviations are model parameters,
not metrology. `reports/calibration-target-analysis-2026-10-29.json` and
`scripts/analyze_calibration_target.py`. No threshold/model/active-file change.
Next: verify horizontal AND vertical square spacing and board/hinge flatness;
do not repeat identical captures or treat released-board fit as acceptance.

Repeated live calibration attempts rejected; latest post-mount-adjustment fit
had 15 views/9 cells but estimated 0.838 m baseline, K focal lengths 979/2373 px,
and 16.65/94.88 px held-out mean/max. No active calibration accepted. Earlier
failures varied widely; latest rejected NPZ has no observations to replay.

Confirmed code defects: independent square-grid orientation switches at 45°
can mismatch physical corners between cameras; old capture pairing could reuse
one source or return stale frames. Added joint grid-symmetry correspondence
with ambiguous/distorted-match rejection (upright co-directed views assumed),
bounded fresh stereo histories requiring both sequences to advance, and
per-capture numeric observation checkpoints. `diagnose_stereo_observations.py`
replays observations without cameras, pruning, or activation. Fixed NumPy
translation-scalar deprecation. Final quality thresholds unchanged.

Verification: 408 tests pass with GPU access, including known-rig diagonal
board/relative-roll regression and stale/reused/skewed pairing tests. Synthetic
20-view replay recovered 100.984 mm for true 101 mm, stereo RMS .085 px,
held-out mean/max .069/.278 px. `reports/calibration-code-check-2026-10-29.json`.
This is software evidence only; not proof these defects explain every live
failure. Next live session needs real board tilts/distance variation, fixed
mounts, and rechecked lens spacing after mount adjustment. No walking acceptance.

Last audited: 2026-10-21. This file is the implementation source of truth.

## 2026-10-28 device follow-up

- Guided calibration attempt collected only 2/15 views. User found repeated
  hold-still prompts without captures difficult. Found hidden pre-nine-view
  coverage-cell veto inconsistent with coach READY guidance. Removed that
  veto: diverse scale/angle views can share cells; completion and final gate
  still require >=70% grid coverage, full quality metrics and >=15 views.
  Capture readiness now checks visibility, size, diversity, both-camera
  steadiness and cooldown; prompts explain blockers, SPACE/c honor readiness.
  Collection no longer auto-finishes at 6/9 coverage (below final gate).
  Board: confirmed chessboard, assumed 7x7 inner corners, measured 24 mm
  squares, user lens baseline 100–102 mm (101 mm midpoint used). No accepted
  calibration yet; metric speech/routes still gated.

- User accepted current OCR as good enough to move on; formal reading CER
  and controls acceptance remain incomplete. Fresh 20 s stationary profile:
  baseline no outages, OCR one 316 ms outage, combined one 389 ms outage;
  combined tick p95 103.9 ms, peak RSS 4073 MiB, minimum available 266 MiB,
  GR3D 99%, OC3 +1413. No operation errors; not T1 acceptance. Prior ~20 s
  outages did not reproduce. Their logs show 203/204 ticks without a stereo
  pair, distinct from shorter detector delays. Root cause still unresolved.
  `reports/reliability-stereo-ocr-2026-10-28-summary.json`. Profile finished,
  cameras released; calibration measurements requested from user.

- Added `run --demo --ocr-preview`: a separate local diagnostic window with
  fresh reading-camera view, last captured source image, and masked actual OCR
  output/status from each voice-triggered read. It shares existing readers,
  adds no OCR worker, and stays outside the demo recording and model input.
  Captured source is not the optional grounder's enhanced crop. No accuracy
  improvement claimed; use this to diagnose framing vs recognition.
  Verification: 395 tests pass with GPU access; CLI help and diff checks pass.

- User visually preferred stereo OCR after three live left/wide comparisons.
  Left OCR 2.83/2.52/4.00 s vs wide 4.62/3.89/4.04 s; left recognized
  163/146/180 characters vs wide 108/85/54. Keep left reading source.
  Exact-reference CER is confounded: reference was title-only while captures
  included different amounts of text, and supplied spelling was unverified.
  Wide had lower raw CER on all three trials, so no measured accuracy-win
  claim is justified. No 12 MP trial recorded. Next accuracy comparison needs
  matched title crops and a verified transcription. Evidence:
  `reports/ocr-camera-comparison-2026-10-28.json`.

- All cameras passed USB3/rate checks after left-cable reseating:
  stereo 59.8/59.9 fps, wide 30.0 fps, arrival skew p95 3.42 ms.
  Evidence: `reports/doctor-2026-10-28-2303.json`.
- Human 600.2 s soak FAILED: eight depth outages, three ~20.46 s late in
  the run with camera reopen logs (cause unestablished). Tick median/p95
  43.3/77.3 ms; stack peak 4172 MiB, minimum available 216 MiB, GR3D peak
  99%, OC3 +499. Model healthy; two reading attempts, no page success claim.
  See `reports/soak-human-2026-10-28-continued-summary.json`.
- Reading now uses the left stereo live stream for coaching and OCR in both
  config copies. `camera.reading_source` accepts left/right/wide. Stereo
  reading does not reopen cameras or consume pairs; wide retains 12 MP stills.
  Stereo reading is 960x600; real-page accuracy remains unaccepted.
- Verification: 393 tests pass with GPU access; live 960x600 OCR capture then
  stereo-pair smoke passed (2.70 ms skew). `reports/reading-camera-2026-10-28.json`.
- Wide-camera SLAM requested but NOT implemented. Metric wide tracking needs
  wide intrinsics/distortion and stereo-to-wide calibration. Current VO stays
  stereo and gated on calibration. Lens untouched; routes remain disabled.

Status meanings: **done** means implemented and verified for its stated scope;
**partial** means useful code/evidence exists but acceptance is incomplete;
**broken** means the current path cannot be relied upon; **not started** means no
implementation exists.

| Component | Status | Verified reality / next gate |
|---|---|---|
| Jetson environment | partial | L4T R36.5.2, MAXN_SUPER. Over-current throttling (oc3) occurs under heavy GPU load on the original PSU; `doctor` and benches record the counter. `jetson_clocks --show` unverified (needs the user's sudo) |
| Rig doctor | done | `python3 -m argus doctor` (+ `--snap`, `--skew-test`, `--bandwidth`, `--focus`): all green with 3 cameras on USB3 (`reports/doctor-2026-10-18-2353.json`). Probes reopen a camera that drops off USB |
| Camera identity/transforms | done (re-verify on recable) | Roles bound by hub port: left 2-1.1, right 2-1.2, wide 2-1.4 (2026-10-21, by parallax). Rotations 0 (worn). Stereo 60 fps (`reports/usb-bandwidth-2026-10-02-2112.json`) |
| Stereo mount alignment | partial | Printed bridge; guided `argus align` brought vertical from 118 px to ~±5-15 px, roll ~1 deg. No clean final measurement (last one discarded: rig was being handled) |
| Stereo calibration tool | partial | Spoken `--coach`; held-out gate (mono RMS <= 0.5, stereo <= 0.7, row mean < 0.5 / max < 1 px, tape baseline +/-10%, >= 15 views, >= 70% coverage); k3 fixed. **No calibration file yet**: needs the user (square size, baseline, flat board) |
| GPU stereo depth | partial | CUDA SAD default (7.6 ms median 480x300/64); VPI SGM selectable (13.0 ms), confidence -> UNKNOWN (`reports/depth-backends-2026-10-21.json`). Metric accuracy blocked on calibration |
| Corridor hazards | partial | `argus/hazards.py`: ground plane, body/head bands, head-level rule, drop-off, head-motion gate (deg/s), 3-frame persistence, refractory, hysteresis; 5 ms/tick; 12 ray-cast tests. Speech only with calibration. Test walk pending |
| Approaching person/vehicle | partial | YOLO11n TRT (3.3 ms GPU, 14.4 ms end to end) + ByteTrack-style tracker + range Kalman + TTC hysteresis; 10 tests; live person detected. Speech only with calibration |
| Legacy reflex / TTC zones | partial | `argus/safety.py`; now keeps only the approach-zone rule when the corridor engine is on |
| Fast-loop timing | **partial — human acceptance pending** | Final 600 s automated combined run: CUDA SAD, tick median/p95 43.2/81.3 ms, no >250 ms outages (max heartbeat 202.7 ms), stack RSS peak 4123 MiB, MemAvailable minimum 426 MiB, GR3D 99%, OC3 +51511. Synthetic voice/page; uncalibrated geometry. See `reports/soak-inference-budget-2026-10-21-summary.json`. Human voice/reading and remaining fault checks required |
| Speech arbiter | done (unit) | P0-P4, preemption < 50 ms in test, answer resume / "repeat?" offer, user-speech hold, depth watchdog earcon (rate-limited). Legacy priority suite fixed and green |
| Wake word | partial | "Argus" / "Hey Argus" via VAD + Whisper tiny with a glossary hint (9/9 synthetic, 0.78 s). Real voice: the name recognised every time; bare "Argus" -> "Yes?". False wakes/hour not measured. Trained openWakeWord model not built (`training/hey_argus/`) |
| VAD / STT | partial | Silero endpointing (0.7 s, 15 s cap) works live; Whisper tiny English only; mishearings seen ("Read the" -> "I'm the") |
| Conversation layer | partial | Session memory/preferences/recap, deterministic assistant tools, 20 s follow-ups, bounded background task/cancellation and truthful outcome history built; see `docs/COMPANION.md`. Live conversation/acoustic acceptance pending.  `argus/dialogue.py`: rule routing + Gemma JSON routing (13/14, 0.99 s), context, follow-ups without the wake word, exit, filler, persona. Live voice: chat 1.6 s, describe 2.4-3.0 s. Reading/seat flows not yet voice-tested end to end |
| Reading / OCR | partial | Software additions: bounded best-of-N still option, 180-degree poor-result retry, clear-gutter column order, cancellable coaching/OCR playback. Synthetic checks only; see `docs/ACCEPTANCE.md`.  RapidOCR CPU; locate document -> CLAHE crop -> OCR (1.5 s); light/glare/blur-aware coach; diagnose and retry. Synthetic sign perfect; real book cover failed on optics (fixed-focus lens, backlight). Needs `doctor --focus` + CER test. Bangla not evaluated |
| Sensitive-text privacy | done (unit) | Luhn cards, BD mobiles, OTP/PIN codes masked before logs, screen and Gemma; demo face blur display-only |
| Face privacy gate | partial | InsightFace, threshold 0.25 after a 0.27 miss; fail-closed. Adversarial set still needed |
| Gemma 4 agent | partial | QAT UD-Q4_K_XL + vision-only mmproj on GPU, 280-token images; describe ~2.2 s; text-only routing/answers. Qwen3.5-2B bake-off not done |
| YOLO-World grounding | partial | TRT FP16 + 401-label precomputed vocabulary (no torch at runtime); optional conf override (reading uses 0.12) |
| Seat guidance | partial | Occupied-seat overlap rejection, matched image/depth snapshots, immediate lost-target range/arrival reset and exception-safe whitelist cleanup built with synthetic checks.  Closed loop on YOLO11n chair/couch/bench; corridor whitelist; direction only until calibration. Live voice test pending (seen when the user stood up) |
| SLAM / navigation | partial | Stereo VO + bounded sparse keyframes/geometric relocalisation; matched depth/pose occupancy mapping, inflated A*, place memory and guarded route follower built with synthetic tests. Walking routes default off; calibration, real drift/relocalisation and walk acceptance pending. No global pose-graph/GPS claim. `reports/companion-build-2026-10-21.json` |
| Demo view / recording | done | `run --demo [--record]`: POV with display face blur, captions, hazard banner, insets, chips; x264 via ffmpeg thread (1.5 ms/frame); TTS audio muxed; closes only on Esc x2 or "shut down" |
| Session logger / study metrics | done | JSONL with masking; `scripts/summarise_session.py` (alarms/min, lead, task times, latency percentiles, precision/recall vs annotated truth) |
| Memory budget | partial | Production excludes optional torch conversion imports (fresh STT worker peak RSS 690.5 -> 360.0 MiB); final combined peak RSS 4123 MiB. 8 GiB NVMe swap active temporarily, no demonstrated benefit in its isolated comparison. System-wide paging still measured; no blanket no-thrashing claim. `reports/stt-import-memory-2026-10-21.json`, `reports/nvme-swap-comparison-2026-10-21.json` |
| Headless service/soak/fault tests | partial | Automated synthetic 600 s soaks and guided human soak tool exist. `argus/service.py`, user unit and installer built (not installed/enabled); owned-process recovery and failure speech have synthetic coverage. `docs/ACCEPTANCE.md`; boot/audio/crash and human/fault acceptance pending |

## Reproducibility pins

Python additions (2026-10-21, `pip install --no-deps` where noted): `rapidocr_onnxruntime==1.4.4` (--no-deps), `pyclipper`, `shapely`. `faster-whisper` bundles the Silero VAD used by `argus/vad.py`. OCR runs on onnxruntime 1.23.2 CPUExecutionProvider (no GPU provider in this build).

### Platform

- Jetson Linux: `nvidia-l4t-core 36.5.2-20260716114719`.
- TensorRT: `10.3.0.30-1+cuda12.5`; `trtexec` reports v10.3.0.
- Python: 3.10 on ARM64.
- NVIDIA Torch: `2.5.0a0+872d972e41.nv24.8`.
- Torchvision: `0.20.0a0+afc54f7`.
- llama.cpp: `ef8268feee28ae943958049bf3bbab4bda99c0ea`, CUDA arch 87.

### Device artifact SHA-256

| Artifact | SHA-256 |
|---|---|
| `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` (production) | `e531007218dfab990486a5de7676a6932d6ea8dea233d1f698d7c21cf8a16889` |
| `mmproj-gemma4-e2b-vision-f16.gguf` (production, derived by `scripts/strip_mmproj_audio.py`) | `366316f696d52b65e33d69cdc55d8a1b3918164708b18df14949b6556b5559c9` |
| `clip/vocab_embeddings.npy` (401 labels, `scripts/build_vocab_embeddings.py`) | `b2db59b8ee6b20e36b5196558a0f57007f24d4ce3eb5fd1955b9eaa2bd0b7da6` |
| `gemma-4-E2B-it-Q4_K_M.gguf` (previous) | `9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d` |
| `mmproj-gemma4-e2b-f16.gguf` | `140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa` |
| `ocr/en_PP-OCRv3_rec_infer.onnx` (SWHL/RapidOCR; evaluated, not default) | `ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea` |
| `yolo11n_416x640_fp16.engine` (movers; trtexec FP16 on device) | `879c51fc923af746e2a2ad85197ad6ec77b1a6e47f296c67a8f95299bcc029a4` |
| `yolo11n_416x640.onnx` (Ultralytics 8.4.67 export, opset 17) | `8d338f38565b2787ea8b7e8661b81164268eddcb545ef3b2f47d2a983e72b5d7` |
| `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `yolov8s-worldv2.pt` | `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792` |
| `yoloworld_640.onnx` | `8b407bb9f5a206d8290fb20d1a6a7d11bc8788da269ec9a583b786290dba4545` |
| `yoloworld_640_fp16.engine` | `8d6b7649ef87f6e280b3e84f492b7b8d910e4d5318f6e31e64aa0659c9fa1912` |
| `ViT-B-32.pt` | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| `yoloworld_runtime_text_640.onnx` | `7f69826f578b66cc057b4eb81659456145ff3962e295981a51682b318f3123fb` |
| `yoloworld_runtime_text_640_fp16.engine` | `7f0cf0a82c0bc5ee713eb91b7085219263160ef806282a3617748abc13b94bfd` |
| Piper ONNX | `5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f` |
| Piper JSON | `efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0` |

The legacy engine above is fixed COCO-80 and is not used. The runtime-text engine
was exported and built on this Jetson with TensorRT 10.3, FP16, and a 2048 MiB
workspace. See `reports/grounding-runtime-text-2026-08-14.json`.

### Python dependencies

`requirements-jetson.txt` contains exact pins for the audited environment.
PyCUDA `2024.1.2` was compiled on-device and initialized one Orin device. Its
transitive packages are pinned too. Do not substitute a generic PyPI Torch build.

## Latest environment report — 2026-08-14

Command: `python3 -m argus --config config/argus.yaml baseline --output
reports/environment-baseline-2026-08-14.json`.

- 24 required checks passed; production readiness remains false.
- A real CUDA matrix workload completed in 0.208 s and drove `tegrastats` GR3D
  to 99%, proving Jetson GPU execution rather than inferring it from imports.
- Both B0495 stereo cameras, the B0459 wide camera, and USB capture/playback
  audio enumerate. All cameras are attached at 5 Gbit/s behind the same USB 3
  hub/controller; sustained bandwidth is deferred to the camera feature.
- Shared memory: 7,976,845,312 bytes; six zram swap devices active.
- All audited model and engine hashes matched.
- Required failures: `jetson_clocks --show` requires interactive sudo and stereo
  calibration is absent. MAXN_SUPER, CUDA stereo, and GPU grounding pass.
- The agent cannot verify privileged locked-clock state without the user's sudo;
  it must not request, store, or bypass that password.

## Test evidence

- 2026-10-21: `python3 -m pytest -q tests/` -> **295 passed** in ~22 s on the
  Jetson (27 test files, including the formerly hanging speech-priority suite,
  VPI, CUDA and TensorRT-free unit tests). Every new module has tests:
  `test_doctor`, `test_timecode`, `test_align`, `test_calibration_gate`,
  `test_reading`, `test_arbiter`, `test_vad`, `test_hazards`, `test_movers`,
  `test_seat`, `test_dialogue`, `test_wake_stt`, `test_vocab`,
  `test_vpi_stereo`, `test_session_summary`.
- Device findings and measurements: `docs/FINDINGS_2026-10.md`.

## Latest GPU depth report — 2026-08-14

`reports/cuda-stereo-depth-2026-08-14.json` records 60 live-frame runs. Warm
latency was 12.7 ms median and 14.1 ms p95 in 15W mode; sampled GR3D peaked at 95%, with RAM
between 3,985 and 4,114 MB. Pair skew was 1.03 ms median but the initial pair was
369 ms and therefore outside the production 12 ms gate. No calibration exists,
so the report is GPU/performance evidence, not metric-distance acceptance.

## Early end-to-end demo — 2026-08-14

`reports/early-e2e-demo-2026-08-14.json` records the positive manual query
“Find the monitor.” The fast CUDA depth loop stayed alive, the 368 ms initial
pair was rejected, privacy ran first on CPU, TensorRT found the monitor at center,
Gemma produced the final answer, and Piper/Pulse playback completed. Slow-path
stages were 0.118 s privacy, 4.269 s visual Gemma, 15.545 s cold grounding, and
1.141 s final Gemma. Peak RAM was 7,376 MB with substantial swap, so co-residency
and cold-start latency remain explicit blockers to wearable acceptance.

## 2026-10-21 Codex T1 audit

321 synthetic tests pass. No model/engine artifacts changed; SHA-256 pins above remain unchanged. `config/argus.yaml` and `/opt/argus/config/argus.yaml` match. Stereo calibration is still absent. A result older than 250 ms from source capture cannot refresh the depth watchdog or drive rules; cached depth also expires. CPU work is bounded (OpenCV 1, face ONNX Runtime 1/1 without spinning, RapidOCR 1/1, Whisper 2 with one worker). These are mitigations, not T1 acceptance.

Automated 10-minute pre-swap stress (`reports/soak-2026-10-21-summary.json`):
CUDA SAD, tick 42.8/79.8 ms median/p95, 4196 MiB peak stack RSS, 401 MiB
minimum MemAvailable, GR3D peak 100%, OC3 +51587; two outages (519/262 ms).
248.5 MiB swap-in / 692.6 MiB swap-out were measured system-wide. The server
survived with batch 256, but T1's zero-gap and no-thrash gates are not met.
Fardeen declined the human soak. The 8 GiB file at
`/opt/argus/swapfile.nvme-8g` is allocated and formatted, mode 0600; not active
at preparation. See `reports/nvme-swap-preparation-2026-10-21.json` and
`docs/operations/MEMORY_TRIAL.md`. Activation requires Fardeen's sudo command.

## NVMe activation and optional import follow-up (2026-10-21)

Fardeen activated the 8 GiB NVMe swap at priority 0, zram remains priority 5,
swappiness 60. The 600 s paired run had 43.7/80.6 ms median/p95 tick and two
outages (302/574 ms), 4118 MiB peak stack RSS, 264 MiB minimum MemAvailable,
GR3D peak 99%, OC3 +50752. No NVMe swap occupancy observed at five-second
intervals. There is no demonstrated timing benefit; T1 remains partial.
Evidence: `reports/nvme-swap-comparison-2026-10-21.json`.

Production startup now excludes optional torch imports. Installed CTranslate2
4.8.0 had loaded torch through conversion/spec modules. Fresh-process Whisper
tiny int8 CPU + Silero ONNX CPU check: worker RSS peak 690.5 -> 360.0 MiB;
ten synthetic transcripts and VAD silence result identical. Policy STT
989.6/1106.2 ms median/p95, OC delta zero. No package or model files changed;
CTranslate2 4.8.0 is now explicitly pinned in `requirements-jetson.txt`.
Unknown object labels fail closed instead of loading live CLIP. 321 tests pass.
Full combined validation completed; see the result below.

Final automated run (`reports/soak-inference-budget-2026-10-21-summary.json`):
600.2 s, CUDA SAD, 43.2/81.3 ms median/p95 tick, no >250 ms outages, worst
heartbeat 202.7 ms, peak stack RSS 4123 MiB, minimum MemAvailable 426 MiB,
GR3D peak 99%, OC3 +51511. PyTorch absent in all resource samples, no reported
request errors. System-wide paging 103.8 MiB in / 188.2 MiB out; single sequential
trial with desktop/warm-cache confounds. Timing and RSS automated subtargets
passed; **T1 remains partial** pending the human workload and fault acceptance.
All model/engine SHA pins remain unchanged. 321 tests pass.


## Deferred-acceptance software build — 2026-10-21

Fardeen requested building now and physical testing later. New runtime changes
have **not** repeated the prior automated timing soak. `docs/ACCEPTANCE.md`
contains the runbook. `reports/build-2026-10-21.json` records synthetic results;
343 tests pass. All model/engine pins and both configuration files are unchanged.

Built: serialized TensorRT execution with owned output arrays; non-consuming
matched stereo observations for seats; occupied-seat rejection and lost-target
cleanup; reading cancellation, bounded chunks, conservative columns, 180-degree
retry and bounded still selection; headless supervisor/user-unit installer;
reading/seat/wake/demo scorers; guided protocol rehearsal and 30-frame privacy
collection; custom VLM variants and bounded frame loading with sensitive masking;
optional local Tesseract Bangla evaluator. No package/model download, service
activation, calibration bypass, or live walking claim. PaddleOCR-VL GGUF mode,
optional trained wake model and any model switch remain evaluation-dependent.
All prototype acceptance requirements remain partial until measured on device.


## Conversational companion / local mapping build — 2026-10-21

386 synthetic tests pass. `reports/companion-build-2026-10-21.json` records build
verification; `docs/COMPANION.md` lists commands and limits. Fardeen asked for
Jarvis-style conversation plus detection, SLAM and assistance. Added bounded
session memory, preferences/name/recap, timers, follow-up listening during
background computation, cancellation-effect suppression, deterministic object
inventory/search answers, local sparse keyframe relocalisation, occupancy/map
planning and place commands. Added bounded masked document excerpts and OCR
redaction of sensitive numbers before visual model requests. All telemetry text
boundaries mask private numbers. Insufficient depth now reports UNKNOWN rather
than CLEAR, and wide frames older than 500 ms are unavailable.

`speech.barge_in_enabled: false` and `navigation.guided_routes_enabled: false`
are explicit in both identical config copies. Both features have software paths
but require acoustic/walk acceptance before enabling. Existing model/engine
pins are unchanged. No new live backend latency, RSS/MemAvailable, GR3D or OC
measurement was taken. The prior T1 soak predates these changes; repeat it,
including calibrated map/VO workloads. All live prototype DoDs remain partial.

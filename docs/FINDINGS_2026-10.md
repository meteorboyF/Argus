# ARGUS findings, October 2026 (device sessions 10-02, 10-18, 10-21)

What was learned on the real rig, with the evidence behind each item. If a
finding conflicts with older documentation, this file is newer; `STATUS.md`
is the per-component truth. Each section ends with **Do / Don't** rules for
whoever works next.

Hardware, unchanged: Jetson Orin Nano Super 8 GB (L4T R36.5.2, JetPack 6,
MAXN_SUPER), 2x Arducam B0495 (AR0234, USB3, stereo), 1x Arducam B0459
(IMX477, USB3, "wide"), USB headset (GeneralPlus USB Audio, mic + phones),
original 19 V barrel PSU, a pink 3D-printed bridge holding the two stereo
cameras on knob-and-bolt heads with the wide camera on a centre post.

---

## 1. Rig, cameras and USB

| Finding | Evidence |
|---|---|
| Camera roles must be bound by **hub port** and re-verified after every recable. They changed three times (10-02, 10-18, 10-21). Current: left `2-1.1`, right `2-1.2`, wide `2-1.4` (`config/argus.yaml`). | `reports/doctor-*.json`, `DECISION_LOG.md` |
| Left vs right is decided by **parallax, not labels**: near objects shift further right in the LEFT image. Moving cameras into the bridge swapped them once (near keychain shifted 219 px more than the far monitor, in the swapped direction). | `DECISION_LOG.md` 2026-10-18 |
| "Connected" ≠ enumerated. With no kernel USB event at all, the cable has no contact; `-71` errors plus a reset mean a strained plug. A B0495 dropped off USB while its knob was being turned. | `journalctl -k` excerpts in `DECISION_LOG.md` |
| USB bandwidth is **not** a constraint. The stereo pair holds 79 fps even with the wide camera at 1080p30; the wide camera reopens in 0.47 s. | `reports/usb-bandwidth-2026-10-02-2112.json` |
| Stereo runs at **60 fps**: capture CPU is 73 / 111 / 152 % of one core at 30 / 60 / 80 fps; worst-case pair skew is 8.3 ms. | same + `DECISION_LOG.md` |
| All three cameras are upright when worn: rotations are 0. | `reports/worn-orientation-2026-10-02.json` |
| B0495 modes: YUYV 960x600 @ 15/30/60/80, 1920x1200 @ 15/30/50. B0459 modes: YUYV 1280x720, 1920x1080, 2016x1520, 3840x2160, 4048x3040. A 4048x3040 still takes ~1.25 s (open + 2 settle frames) at 6.4 fps; the 1080p stream is restored afterwards. | `v4l2-ctl`, `reports/reading-latency-2026-10-21.json` |
| **The B0459 has no focus control** (fixed, hand-focused lens) and no exposure compensation. As shipped it focuses far: a page at arm's length is blurred (Laplacian 47). A bright window behind a page makes it read as 46/255 against a 126 background. | `reports/reading-latency-2026-10-21.json`, `DECISION_LOG.md` 2026-10-21 |
| The board throttles on over-current under heavy GPU load in MAXN_SUPER (`oc3` channel, 700–965 events per Gemma benchmark), even on the original PSU. It is protective, not damaging, but GPU latencies measured under it are pessimistic. | `argus doctor` row, `reports/vlm-m3-qat-2026-10-02.json` (`oc_throttle_events`) |

**Do:**
- run `python3 -m argus doctor` after any physical change;
- check the left/right binding by parallax;
- tighten both screw-locks and give each cable a slack loop tied to the bridge.

**Don't:**
- trust `/dev/videoN` numbers or port hints after recabling;
- run GPU benchmarks without recording the OC counter (`doctor.oc_events()`).

## 2. Stereo alignment and calibration (still open)

| Finding | Evidence |
|---|---|
| The old ball-head mount was toed out ~25° (worn, at 1 m: 328 px disparity, 13.6 px vertical, ~1° roll). | `reports/worn-orientation-2026-10-02.json` |
| The printed bridge fixed most of it. After a guided session: vertical within about ±5–15 px, roll "very slightly", overlap 81–91%. A clean final measurement was never taken: the last one was done while the rig was being handled and was discarded. | `reports/align-*.json`, `DECISION_LOG.md` 2026-10-21 |
| Alignment tolerances must sit above estimator noise and what a hand can hold. 3 px / 0.5° / 1° looped forever; now 8 px / 1° / 2°, with 1.5x hysteresis while locking. | `argus/align.py` `Targets` |
| Yaw from a homography on a near scene reads the baseline parallax as toe-out (~7° at 1 m for 13 cm). Measure yaw on a scene ≥ 3–4 m away. A near person in view corrupts homography pitch: size pitch from vertical pixels. | `argus/align.py` |
| Never measure with the camera facing its own monitor. Video feedback gave 775 fake matches and a perfect-looking 0 px. | `DECISION_LOG.md` 2026-10-18 |
| Calibration gate: OpenCV's free k3 overfits held-out views badly (0.1 px corner noise → 2.3 px worst row error; 0.2 px → 16 px). k3 is now fixed at 0 (0.65 / 1.37 px). The gate requires a tape-measured baseline. | `tests/test_calibration_gate.py`, `DECISION_LOG.md` |
| **There is still no `/opt/argus/config/stereo_calib.npz`.** Everything metric (hazard speech, approach TTC speech, seat steps, VO, corridor guidance) is coded but stays off by design until it exists. | `STATUS.md` |

**Do:**
- before calibrating, get from the user: the square size (measure across 4 squares, divide by 4, in mm), the lens-to-lens baseline in mm, and confirmation that the board lies flat;
- then run `scripts/calibrate_stereo.py --coach --square-mm N --baseline-mm B`, then `--verify`.

**Don't:**
- relax the calibration gate to get a file saved;
- enable metric speech without a gated calibration.

## 3. Memory and models

| Finding | Evidence |
|---|---|
| The Gemma mmproj bundled a 612 MB **audio** tower. A vision-only projector is 373 MB (`scripts/strip_mmproj_audio.py`). | `reports/vlm-m3-steps-2026-10-02.json` |
| Production Gemma: **QAT UD-Q4_K_XL**, vision mmproj **on GPU**, `--image-max-tokens 560 -ub 1024 -b 1024`, ctx 2048, `--cache-ram 0`, one slot, flash-attn. Describe ~2.2 s, find ~1.7 s, read-560 ~3.5 s warm (all under OC throttling). | `reports/vlm-m3-qat-2026-10-02.json`, `scripts/run_llama_server.sh` |
| A 256 px image cap made Gemma answer "I cannot see… I am a text-based AI". Images are now sized by token budget (48x48 px per token, 280 tokens). | `DECISION_LOG.md` 2026-10-02 |
| Benchmarks must send `cache_prompt: false`, or repeated images report prompt-cache speeds. | `scripts/bench_vlm.py` |
| Grounding no longer loads torch: a 401-label CLIP vocabulary is precomputed (`config/vocab_en.txt` → `/opt/argus/models/clip/vocab_embeddings.npy`). RSS 1672 → 353 MB, warm-up 11.6 → 1.6 s. Torch/CLIP load only for out-of-vocabulary words. | `reports/mem-budget-2026-10-02.json` |
| Per-component RSS: privacy 173, grounding 353, CUDA SAD 120, Piper 157, Whisper tiny 555, openWakeWord 200 MB, llama-server ~3.1 GB. Projected stack ~4.6 GB. **Not yet soak-measured.** On the dev desktop VS Code + Chromium + agents hold ~3 GB more. | `reports/mem-budget-2026-10-02.json` |
| The Qwen3.5-2B bake-off has not been done. | — |

**Do:** pin SHA-256 for every new model in `STATUS.md`, and measure RSS and `MemAvailable` together.

**Don't:**
- reintroduce torch into the runtime path;
- use `--mlock`;
- switch to E4B.

## 4. Perception modules

| Finding | Evidence |
|---|---|
| Depth backends on 40 live, **unrectified** pairs: CUDA SAD 480x300/64 is 7.6 ms median; VPI SGM 480x300/64 is 13.0 ms (denser: 94–98% valid on synthetic, against SAD's 85%); SGM 960x600 is 30 ms and too heavy next to Gemma. SAD remains the default; `vpi_sgm` is selectable. Choose on calibrated tape accuracy. | `reports/depth-backends-2026-10-21.json` |
| Synthetic stereo tests need the real sign convention: a point appears further LEFT in the right image. A wrong-sign pair made VPI look broken. | `tests/test_vpi_stereo.py` |
| Corridor engine (`argus/hazards.py`): RANSAC ground fitted on the **bottom image rows** (fitting all low points let a lower floor past a step win and hid the drop); vectorised, 5 ms per tick. A head-worn camera looks down 20–25°: a 10° synthetic camera could not see the floor before 2 m. The head-motion gate is in deg/s via phase correlation (Farneback failed on fine texture). | `tests/test_hazards.py`, `DECISION_LOG.md` |
| Movers: YOLO11n FP16 TensorRT at 640x416, **3.3 ms GPU, 14.4 ms end to end** after `cv2.dnn.blobFromImage` (numpy pre-processing cost 15 ms). Runs on the (rectified) left frame so boxes align with depth. Found a live person at 0.91. | `argus/movers.py`, `STATUS.md` pins |
| **pycuda contexts are per thread.** A TensorRT engine created on a worker thread failed ("no currently active context"). `TRTRunner` now pushes the primary context per call. | `argus/trt_runner.py`, `DECISION_LOG.md` |
| The grounder finds a hand-held book at only 0.19 confidence: reading locates documents at 0.12 (a wrong crop only costs time). | `argus/orchestrator.py` reader construction |
| Privacy: a dark, side-on face scored 0.27 and passed the 0.5 default; the threshold is now 0.25. Demo/recording face blur is display-only and pads boxes upward. | `DECISION_LOG.md` 2026-10-02 |

## 5. Reading (OCR)

| Finding | Evidence |
|---|---|
| RapidOCR 1.4.4 (PP-OCRv4 ch det/rec, onnxruntime **CPU**; onnxruntime has no GPU provider here). A synthetic sign reads perfectly in ~1.2 s. RapidOCR resizes to `max_side_len 2000`, so a full 12 MP still loses detail: **locate and crop first**. | `reports/reading-latency-2026-10-21.json` |
| The default recogniser is Chinese+English: it produced a lone "米" on a textless scene, so CJK-only tokens are dropped. The English PP-OCRv3 recogniser (`/opt/argus/models/ocr/en_PP-OCRv3_rec_infer.onnx`) was **not** better on the blurred cover. | `argus/ocr.py`, `DECISION_LOG.md` |
| A real book cover (*Crime and Punishment*): raw still → no text; crop + CLAHE on lightness → "Ght and Pnishment" (0.89). The root causes are **optical** (fixed focus, backlight); no model fixes them. | `DECISION_LOG.md` 2026-10-21 |
| The coach now checks light before framing (backlit / dark / glare), uses the book finder when no words are detected yet ("I can see it, but it's blurry. Move it a bit back so I can see it properly"), and diagnoses and retries failed reads twice. On the real failed still the diagnosis is exactly the backlight. | `tests/test_reading.py` |
| Gemma is used only for questions **about** OCR text, as masked text, never on the image. | `argus/agent.py` `ask_text` |

**Open:**
- the lens refocus (`python3 -m argus doctor --focus`, with the user);
- a book-cover and printed-page CER test (DoD: CER < 3%, < 4 s to first word);
- the Bangla evaluation (Tesseract `ben` needs apt/sudo; PaddleOCR-VL-Bangla GGUF memory test).

## 6. Speech, conversation and wake word

| Finding | Evidence |
|---|---|
| After a reboot PulseAudio fell back to the onboard analog jack: the mic recorded pure zeros and speech went nowhere. `prefer_usb_audio()` selects the USB headset at every start. | `argus/speech.py` |
| The arbiter (`argus/arbiter.py`): P0 DANGER cuts speech (< 50 ms in test), P1 WARN cuts P2–P4, interrupted answers resume if < 5 s old, user speech holds P2–P4. The old speech-priority "hang" was a test race (the fake interrupted before the queue was filtered); fixed, and the suite runs green. | `tests/test_arbiter.py`, `tests/test_speech_priority.py` |
| Silero VAD (bundled with faster-whisper): endpoint after 0.7 s silence, 15 s cap. While waiting, only the pre-roll is kept (no unbounded buffer). | `tests/test_vad.py` |
| Wake word = the spoken name via VAD + Whisper tiny with a **glossary hint** "Names: Argus." (9/9 on synthetic phrases, 0.78 s). A prefix hint ("Argus, hey Argus.") made Whisper *drop* the name from longer phrases. Whisper base was no better and slower (1.4 s, 41 s load). | `argus/wake_stt.py`, `DECISION_LOG.md` |
| Real-voice test: the name was recognised every time, but a bare "Argus" answered only with a beep went unnoticed. Now it answers "Yes?" and listens 8 s, **while** saying "Yes?". Every heard phrase is logged as `heard` in the session JSONL. | `reports/sessions/2026-10-21-125728.jsonl` |
| Conversation layer (`argus/dialogue.py`): rules first (control, read, next page, seat, find, describe, exit, filler), then one Gemma text-only call returning validated `{skill, name, say}`. Live routing 13/14 at 0.99 s median. Gemma once claimed "I am closing the first person view now"; the persona now forbids claiming untooled actions, and `shut down` is a real command. | `reports/dialogue-routing-2026-10-21.json` |
| Questions about printed text ("what is the name of the book?", "who wrote this", anything with title/author/writer) must go to **reading**, not chat or find. Gemma had routed the title question to `find`, and Whisper misheard "Read the title" as "I'm the title". | `tests/test_dialogue.py` |
| Real voice turn times: chat ~1.6 s, describe 2.4–3.0 s (privacy 0.26–0.6 s + Gemma ~2.2 s). | `reports/sessions/2026-10-21-124310-summary.json` |

## 7. Demo, logging and robustness

| Finding | Evidence |
|---|---|
| The demo window must not end a session by accident: losing focus closed it once, and a stray Esc twice. Now it closes only on **Esc twice within 2 s** or "Argus, shut down"; the close reason is logged. The dashboard had shadowed `Thread._stop` with an Event, which crashed `join()`. | `argus/dashboard.py` |
| Recording: the Orin Nano has **no hardware video encoder**; 1080p MJPG cost 138 ms per frame. Frames are downscaled to 960x540 and piped from a thread to ffmpeg x264 ultrafast (1.5 ms per frame in the render thread). TTS and earcon PCM are tapped to an audio timeline. | `argus/demo_view.py` |
| Never use `pkill -f <pattern>` where the pattern appears in your own command line: it killed the agent's shell twice and lost a run's output and recording. Stop processes by PID (`pgrep -f "^python3 -m argus ..."` then `kill -INT`). | — |
| The skew test once **OOM-killed** the board (and a VS Code process) by buffering 4000 raw frames per camera. Measurement tools must buffer results, not images. | `DECISION_LOG.md` 2026-10-02 |
| **Open safety issue:** with the full demo stack (Gemma, Whisper, demo view, face blur, recording) the fast loop's depth heartbeat had gaps of 0.26–0.73 s several times a minute, and skew drops rose to 12–20 ms. The cause is not yet found: suspect CPU/GIL contention. Must be fixed before any walking test. | `reports/sessions/2026-10-21-124310.jsonl` (`depth_gap` events) |

## 8. Codex T1 continuation, 2026-10-21 (acceptance incomplete)

| Finding | Evidence |
|---|---|
| Doctor: stereo 59.8/59.9 fps, arrival skew median/p95 2.23/3.31 ms; all cameras on expected USB3 ports. RAM check failed: 1403 MiB available, 2156 MiB swap used. Snapshots were inspected: real room/ceiling, unsuitable for geometry acceptance. | `reports/doctor-2026-10-21-1336.json` |
| Latest-only stereo selection rejected 67/202 baseline ticks through the unchanged skew gate. Three-frame pairing with both sequences advancing produced 301/301 fresh ticks in a later 30 s stationary check. This measures arrival-time pairing, not exposure sync or calibration accuracy. | `reports/t1-profile-default-2026-10-21.json`, `reports/t1-profile-budgeted-2026-10-21.json` (latter had no llama-server) |
| Default CPU pools plus busy-page OCR: tick median/p95/max 80.9/138.0/365.6 ms; display publication reached 236 ms. Limiting pools reduced OCR-phase p95 to 78.0 ms, but did not establish absence of tails. | `reports/t1-profile-default-2026-10-21.json`, `reports/t1-profile-budgeted-live-2026-10-21.json` |
| llama-server twice exited with failed compute-prefill buffer allocation. The second log names a 234.04 MiB `cudaMalloc` failure. No kernel OOM kill found. Reduce batch/microbatch 1024 to 256 to bound transient memory; this is distinct from changing models or allowing CPU depth. | `reports/t1-failures-2026-10-21.json` |
| Moving previews off the fast thread reduced baseline publication median from 16.7 to 0.16 ms. With CPU caps, pairing, preview worker and batch 256, the short combined phase tick median/p95 was 34.5/56.7 ms; RSS peak 3856 MiB; MemAvailable minimum 292 MiB; GR3D peak 99%; OC3 +1654. A separate OCR phase still had an outage over 250 ms. | `reports/t1-profile-final-2026-10-21.json` |
| USB `-71` errors and an OpenCV video2 select timeout occurred during the budgeted matrix. The left reader reopened. Freshness pairing withheld stale frames; recovery latency is still unacceptable for uninterrupted operation. | `reports/t1-failures-2026-10-21.json` |
| A recovered `depth_gap` event is insufficient evidence: a camera can remain unavailable at the end of a phase. In the budgeted-live faceblur phase only 66/201 ticks had fresh depth. Reports must include open outages, not just completed heartbeat percentiles. | `reports/t1-profile-budgeted-live-2026-10-21.json`, `tests/test_timing.py` |
| NVMe root is ext4, 399 GiB free. Initial swap is six zram devices, 3803 MiB total, priority 5; no NVMe swap. Zram data occupied about 2.8 GiB logical / 970 MiB physical during the inspection. Additional NVMe swap is a pending controlled trial, not a latency fix demonstrated yet. | `reports/memory-before-nvme-swap-2026-10-21.json` |

**Do:** retain fail-closed gates and the real watchdog; count terminal outages;
review swap-in/out and MemAvailable alongside RSS. The busy synthetic OCR page
has roughly 42 lines and is deliberately heavier than the earlier synthetic
sign, so their OCR latency numbers are not an accuracy or speed comparison.

**Don't:** call the short matrix a 10-minute human soak, compare cached Gemma
request speeds, or treat the interrupted/no-server runs as full-stack evidence.
No reading CER, calibrated hazard, seat, or walking acceptance was completed.

### Ten-minute automated pre-swap outcome

`reports/soak-2026-10-21-summary.json` (details in
`reports/soak-automated-offscreen-2026-10-21.json`): 600.6 s, CUDA SAD,
5934 fresh depth results / 5935 ticks, tick median/p95 42.8/79.8 ms,
peak stack RSS 4196 MiB, minimum MemAvailable 401 MiB, GR3D peak 100%,
OC3 +51587. Two measured depth outages were 518.8 and 261.9 ms. No model
request errors; llama batch 256 survived. System-wide paging: 248.5 MiB in,
692.6 MiB out, 67065 major faults. T1 fails despite meeting the p95 and RSS
subtargets. Synthetic speech/page, offscreen render/encode, uncalibrated rig;
human voice/reading acceptance is still absent. A small synthetic test run
overlapped the tail, and the final post-mover age guard needs validation in
the post-swap run. The prepared NVMe swap file is not evidence of an
improvement until activated and measured (`reports/nvme-swap-preparation-2026-10-21.json`).

### NVMe activation and optional-framework follow-up

Fardeen activated the 8 GiB swap file (root-owned, mode 0600, priority 0),
leaving priority-5 zram and swappiness 60. A further 600 s combined run used
CUDA SAD and measured 43.7/80.6 ms median/p95 tick, 4118 MiB peak RSS,
264 MiB minimum MemAvailable, GR3D peak 99%, OC3 +50752; two outages
301.5/573.6 ms. No NVMe occupancy was observed in five-second samples.
Activation did not demonstrate a timing benefit. System-wide swap-in/out
was 260.8/1176.7 MiB (`reports/nvme-swap-comparison-2026-10-21.json`).

The running process mapped torch libraries despite TRT grounding. Installed
CTranslate2 4.8.0 imports torch through optional conversion/spec modules.
Excluding that optional import in fresh workers reduced peak RSS
690.5 -> 360.0 MiB. Ten synthetic Whisper transcripts were identical;
Silero silence probability was identical (0.0238286). Policy STT median/p95
989.6/1106.2 ms, minimum MemAvailable 1748.8 MiB, OC delta zero. STT/VAD
backends were CPU; GR3D peak 74% reflected the desktop, not GPU STT.
`reports/stt-import-memory-2026-10-21.json` records both workers. Do not
attribute this memory saving to swap or claim real-voice acceptance.

### Final automated run with production import budget

`reports/soak-inference-budget-2026-10-21-summary.json` (full samples in
`reports/soak-inference-budget-2026-10-21.json`): 600.2 s combined synthetic
load, CUDA SAD, tick median/p95 43.2/81.3 ms, no >250 ms depth outages,
worst heartbeat interval 202.7 ms, peak stack RSS 4123 MiB, minimum MemAvailable
426 MiB, peak GR3D 99%, OC3 +51511. Torch was absent from all sampled states;
process maps were checked during the run and contained no torch libraries.
No reported request errors. Completed operations: 189 Gemma, 253 Whisper,
27 busy-page OCR, 7282 rendered frames. Latency distributions are in the summary.

System-wide swap-in/out 103.8/188.2 MiB and major-fault delta 27786 were lower
than the preceding run, but the desktop and warm caches prevent a causal
latency guarantee. Do not claim a full-stack RSS reduction from the isolated
330.5 MiB worker saving: peak combined RSS was comparable across these trials.
Automated timing/RSS subtargets passed once; human continuous queries/reading,
calibrated geometry and full fault acceptance are still pending. NVMe swap
remains temporarily active, with zero occupancy at the final inspection;
its earlier isolated trial did not demonstrate a timing benefit.


## 2026-10-21 — Deferred-testing build note (no new device measurements)

Fardeen requested software construction now and physical testing later.
Static inspection identified competing seat/stereo-pair consumption and a
shared TensorRT execution context/buffer race. Both paths are changed and
covered synthetically; this is a code finding, not measured device behavior.
Seat stale-range/arrival cleanup and reading stop behavior are also covered.
Evidence: `reports/build-2026-10-21.json`; procedures: `docs/ACCEPTANCE.md`.
No new backend latency, RSS/MemAvailable, GR3D or OC measurements were taken.
The previous soak must be repeated on the new build; all live DoDs stay partial.


## 2026-10-21 — Companion/map build scope (no new live findings)

Code inspection found insufficient legacy depth returned CLEAR despite the
comment describing uncertainty. It now returns UNKNOWN, with a synthetic
regression retaining the original scenario. Wide frames also expire. New
session assistant, sparse keyframe/map and OCR privacy workloads are built;
`reports/companion-build-2026-10-21.json` records synthetic checks only. There
are no new device latency/RSS/MemAvailable/GR3D/OC or acoustic measurements.
Repeat the full-stack soak on this build, especially after calibration enables
mapping, before attributing any reliability to these additions. Commands and
limitations: `docs/COMPANION.md`. Do not infer global SLAM/walking acceptance.

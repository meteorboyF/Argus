# ARGUS architecture

`STATUS.md` is authoritative for what works today. This file describes the
design every module follows. Device measurements are in
`docs/FINDINGS_2026-10.md`.

## Principles

- **Two speeds.** Safety runs on-device, fast, deterministic and terse
  (earcon + at most 3 words). Generative AI is on demand and separate.
- **Configure down, inhibit up.** The slow loop may *configure* the fast loop
  (for example, whitelist the seat being approached for body-band speech) but
  can never disable it. The fast loop *inhibits* slow-loop audio: hazards
  preempt any speech.
- **Gemma talks; the skills do the work.** A language model routes requests
  and phrases replies; deterministic code reads, measures and decides
  anything safety-relevant.
- **Fail closed.** Missing calibration means no metric speech; a missing
  privacy gate means no image to Gemma; a missing GPU backend means no
  production start. Invalid or stale depth is UNKNOWN, never CLEAR.
- **Guide, don't fail.** When perception is not good enough, tell the person
  what to change ("move it a bit back so I can see it properly").

## Fast safety loop (10 Hz, its own thread)

```text
AR0234 left/right (60 fps, per-camera reader threads, arrival timestamps)
  -> skew gate (12 ms)
  -> rectification (only with stereo_calib.npz)
  -> GPU disparity: CUDA SAD (default) | VPI SGM (confidence -> UNKNOWN)
  -> hazards.py: RANSAC ground plane (bottom rows), corridor 0.8 x 2.0 m,
       body / head bands, drop-off, head-motion gate (deg/s), alarm hygiene
  -> movers.py: YOLO11n TRT on the left frame -> tracker -> range Kalman -> TTC
  -> safety.py: legacy approach-zone rule
  -> arbiter: earcon + "Stop. Left." (P0/P1), heartbeat for the depth watchdog
```

Metric speech requires calibration. Detection, tracking and depth are shown on
the dashboard regardless.

## Slow conversational loop

```text
mic (USB headset, Pulse default forced) -> Silero VAD phrases
  -> wake_stt.py: Whisper tiny + "Names: Argus." hint; phrase starts with the name?
       "Argus, <request>" -> request     "Argus" -> "Yes?" -> next phrase
  -> dialogue.py
       rules: stop/pause/continue/repeat, next page, read, text questions
              (title/author/label/price/date), seat, find, describe, exit, filler
       else Gemma text-only router -> {skill, name, say} (validated)
  -> skills
       describe      -> wide frame -> PRIVACY GATE -> Gemma (280-token image)
       find          -> YOLO-World TRT (precomputed vocabulary) -> direction
       read          -> reading.py coach (light / glare / blur / position, book finder)
                        -> 12 MP still -> locate document -> CLAHE crop -> RapidOCR
                        -> sentence playback | answer just the question (Gemma, masked text)
       page_question -> Gemma on masked OCR text
       seat          -> seat.py closed loop (re-detect every 0.7 s, corridor whitelist)
       chat          -> Gemma answer (persona: short, never claims untooled actions)
       exit          -> "Closing. Goodbye." -> clean shutdown
  -> arbiter.py (P2 guidance / P3 answers / P4 system) -> Piper (paplay)
  -> follow-ups for 6 s without the wake word after ARGUS asks something
```

## Resource ownership

- **GPU:** Gemma (llama.cpp, QAT Q4 + vision projector), YOLO-World TRT,
  YOLO11n TRT, stereo disparity.
- **CPU:**
  - capture threads, rules, tracker, Kalman,
  - InsightFace privacy,
  - RapidOCR, Silero, Whisper, Piper,
  - dashboard and recording (ffmpeg x264).
- **Memory:** the target is a resident stack ≤ 5.5 GB. No torch at runtime
  (precomputed CLIP vocabulary). TensorRT engines run via `TRTRunner`, which
  pushes the CUDA context per call so any thread may use them.

## Modules

| Module | Responsibility |
|---|---|
| `cameras` | discovery, hub-port role binding, reader threads, `capture_still()` |
| `depth`, `cuda_stereo`, `vpi_stereo`, `calib_health` | rectified GPU disparity → metric depth; drift monitor |
| `hazards` | corridor engine (ground, bands, drop, motion gate, alarms) |
| `movers` | YOLO11n detector, tracker, approach TTC |
| `safety` | legacy reflex (approach zones) |
| `slam`, `navigation`, `seat` | VO, corridor guidance, seat guidance |
| `grounding`, `vocab`, `trt_runner` | open-vocabulary find, label embeddings, TensorRT |
| `privacy`, `sensitive` | face blur gate; text masking |
| `ocr`, `reading` | OCR pipeline; reading session and coach |
| `agent`, `dialogue` | Gemma client (image, text, router); conversation manager |
| `speech`, `arbiter`, `vad`, `wake_stt`, `earcons` | TTS, priorities, endpointing, wake on name, cues |
| `doctor`, `align`, `guide`, `timecode` | rig tools; spoken + on-screen guidance for humans |
| `telemetry`, `dashboard`, `demo_view` | bus + `SessionLogger`; monitor views; recording |
| `orchestrator` | lifecycle and wiring only: new behaviour goes into modules |

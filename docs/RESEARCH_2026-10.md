# Research conclusions (October 2026), summary

Copied from section 2 of JETSON_PROMPT_03 (the full sourced research document
was never committed). These are the design targets the code follows; measured
results that refine them are in `docs/FINDINGS_2026-10.md`.

## Architecture
- The Two-Speed split matches shipped BLV products (Biped NOA, .lumen, Glide,
  WOAD): safety on-device, fast, rule-based, terse, earcon-first; generative
  AI on demand and separate.
- "Configure down, inhibit up": the slow loop may configure the fast loop but
  never disable it; the fast loop inhibits slow-loop audio. Shared state is a
  timestamped blackboard. TTLs: depth 300 ms, detector 1 s, VLM ~5 s.

## VLM
- Stay on Gemma 4 E2B; not E4B (Q4_K_M 4.98 GB + 0.99 GB mmproj, 11–17 tok/s,
  +3 OCRBench).
- Use QAT UD-Q4_K_XL; use a vision-only projector on the GPU.
  `--image-max-tokens` ~280 for describe and 560 for read. `--cache-ram 0`,
  `--parallel 1`, flash-attn, mmap; no `--mlock`. (All done.)
- Challenger: Qwen3.5-2B (better OCR, weaker instruction following); decide
  by a measured bake-off.

## Reading
- Small VLMs give confident misreads on text: dedicated OCR for verbatim
  text, the VLM only for questions about it.
- English: PP-OCRv5 mobile via RapidOCR (v4 currently used).
- Bangla: no TensorRT OCR. Candidates are PaddleOCR-VL-1.6-Bangla GGUF and
  Tesseract `ben` / EasyOCR `bn`.
- Read from a full-resolution still.

## Depth
- The Orin Nano has no PVA/DLA: stereo shares the GPU with the VLM.
- VPI SGM (CUDA) is the expected upgrade over SAD.
- f ≈ 550 px at 960 wide; 128 disparities → ~0.69 m minimum range at
  B 0.16 m.
- Unsynchronised stereo: 30 fps pairs can be 16.7 ms apart, so a 30°/s head
  turn costs ~5 px. Mitigations:
  - highest stable fps,
  - skew gate,
  - 3-frame persistence,
  - an optical-flow head-motion gate,
  - calibrate only from static views.

## Mount
- 0.1° of relative yaw ≈ 1 px disparity bias; mount parallel and rigid.
- Rectified row error target: mean < 0.5 px, max < 1 px.

## Hazard rules (Biped NOA / ImpactAlert evidence)
- RANSAC ground plane, then a corridor 0.8 m wide × 2.0 m tall that catches
  head-level obstacles.
- Drop-off: ground > 10 cm below the plane, or a confident hole.
- Approaching person: detector → tracker → median depth of the central half
  of the box → Kalman → TTC < 3 s with hysteresis. (Raw TTC was 51%
  accurate; hysteresis plus tracking added ~42 points.)
- Alerts: earcon plus ≤ 3 words.
- Latency budget ~0.3 s + 0.4 s speech + 1.5 s reaction → trigger at
  2.5–3 m.

## SLAM
- cuVSLAM needs ±100 µs sync, which is impossible on this rig. Stereo VO is
  experimental, and seat guidance must not depend on it: use closed-loop
  re-detection (0.7 m per step).

## Speech
- Custom "Hey Argus" wake word: ≤ 0.5 false accepts/h at ≥ 80% recall; the
  hey_jarvis weights are CC BY-NC-SA.
- Silero VAD endpointing: 0.6–0.8 s silence, 15 s cap.
- Stock Whisper tiny/base is unusable for Bangla.
- Piper: stream sentence by sentence, keep it warm, keep barge-in.

## Privacy
- Face gate (InsightFace detection-only).
- Sensitive-text masking: Luhn card numbers, BD mobiles
  `(\+?880)?1[3-9]\d{8}`, OTP/PIN keywords.

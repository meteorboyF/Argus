# Feature recording session — 2026-10-29

One separate MP4 and JSONL per feature. These are prototype demonstrations;
record only observed outcomes. ARGUS demo recording includes its own speech
and earcons, captions of recognized requests, camera view, and continuous
audio from the default PulseAudio microphone mixed with ARGUS playback.
Use a headset to reduce acoustic echo. Earlier takes contain ARGUS audio only.

Latest sector calibration was rejected (20 views, stereo RMS 2.299 px,
held-out mean/max 2.101/8.098 px). Keep metric warnings and routes gated.

| Clip | Demonstration | Current scope |
|---|---|---|
| 01-seat | “Argus, where is the seat?” | Left stereo camera; stationary direction finding only; no distance or walking claim |
| 02-incoming | Approaching person/object | Object presence can be demonstrated; metric TTC/danger warnings blocked on accepted calibration |
| 03-mapping | Local mapping/SLAM | Current backend is calibrated stereo VO, currently gated; wide-camera SLAM unimplemented |
| 04-reading | “Read this” then page question | Left-camera OCR; masked page text to Gemma; stop/repeat during speech gaps or typed controls |
| 05-conversation | Natural question and follow-up | Local Gemma E2B; name/preference memory is session-only |
| 06-privacy | Consenting person in scene; scene question | Display face blur and privacy-gated model query; observe actual detection, do not claim perfect anonymization |

## First clip: seat direction

Keep wearer seated or rig stationary, headset on. Place an empty chair in
view of both stereo cameras. Say “Argus, where is the seat?” Listen for a
direction. After ARGUS finishes speaking, say “Argus, stop.” Change the chair
position and repeat to check direction changes. Do not walk on its directions.
Say “Argus, shut down” or press Esc twice to finish and finalize the MP4.
Review the saved clip and log before calling the feature demonstrated.

Use `run --demo --demo-source left --record PATH.mp4 --session-log PATH.jsonl`
for seat/OCR clips. Use `--demo-source wide` for scene/privacy clips. Runtime
privacy settings and calibration gates remain unchanged. `--ocr-preview`
opens a local diagnostic window; that extra window is not in the MP4.

## Redesigned presentation interface

The demo now shows a large camera view, feature title, assistant activity,
and a readable request/response panel. Reading mode includes actual masked
OCR text in the main view, so it is included in recordings. Select a fixed
title with `--demo-feature reading` (or `seat`, `objects`, `hazards`,
`mapping`, `conversation`, `privacy`); the default `auto` follows intent.
Press Tab to toggle technical diagnostics. Without `--record`, the footer
explicitly shows that this is an unrecorded preview. Recordings retain the
existing 960×540 encoding and ARGUS audio behavior.

Offline design examples are in `reports/interface-redesign/`. Their sample
text is illustrative, not evidence of OCR accuracy or a new feature test.

Hidden keyboard control: with the demo window focused 
press R to save the current MP4 and start the next unused take. The existing
session JSONL continues and records recording_reset events identifying clips.
This resets the recording timer and clears typed input and displayed request,
response and OCR text; conversation memory and running tasks remain.

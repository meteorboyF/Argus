# ARGUS agent handoff

## Newest recording checkpoint — 2026-10-29

All runtimes stopped; local Gemma server may still run. User retained reading
take9, conversation take10, privacy take2, seat take3 in
reports/feature-recordings/2026-10-29; matching JSONL and session-state.json.
Reading take9 scene response only; earlier take6 has actual OCR title evidence.
Conversation take10 six-turn exchange,102.633s H264/AAC. Privacy take2 46.998s,
H264/AAC, visual review pending. Seat take3 wide display per user request,
stereo detection unchanged; target loss/directions oscillate. User says seat
needs work, explicitly deferred. No acceptance. Seat take2 also had wide/left
camera stalls. Continuous mic+ARGUS mix now in new takes; streams verified,
voice intelligibility not yet reviewed. Removed distance-guidance unavailable
footer; runtime safety gates unchanged. Next explain metric alerts still
blocked by calibration and wide SLAM unimplemented; remaining useful demos
include general scene description/object finding, or review saved evidence.
Do not restart calibration or seat work automatically. Preserve uncommitted work.


## Latest continuation — microphone recording, 2026-10-29

Runtime stopped; reading takes 3–7 saved, latest 7 verified 89.734s H264/AAC.
All three cameras passed simultaneous 15s doctor: stereo59.9fps,wide30fps,
USB3 all,skew p95 7.21ms (report doctor-2026-10-29-0358.json).
User removed top feature titles and automatic startup scene query. Reading
coach now checks denoised detected-word contrast before rejecting dark covers;
take5 captured, take6 OCR University of Alberta and direct title response.
No overall OCR accuracy acceptance. Removed unsolicited end-of-page speech;
document prompt requests answer only.53reading+50dialogue+5layout tests passed.
User requested microphone in clips: Recorder now captures continuous default
PulseAudio source to disk via FFmpeg and mixes with ARGUS playback. Current
source is Bluetooth headset. Technical 5s smoke succeeded H264/AAC,4.27s nonzero
mic PCM; no human speech intelligibility verification. Report/video named
microphone-mix-check under feature-recordings/2026-10-29. Added timing offset
based on mic duration/stop to compensate FFmpeg startup; not live rechecked.
Previous takes remain ARGUS-only. No new feature recording active. Next start
new take with mic and verify both voices. Preserve all uncommitted changes.


## Newest continuation — 2026-10-29

POWER-OFF HANDOFF: user is switching Jetson off for a while. Resume from
this section, not earlier calibration/recording instructions. Current work:
operational native UI completed (no taglines/suggested commands/demo numbers),
conversation prompt/routing improved and local Gemma launcher restored.
Latest OCR120ms motion fix DID capture live: reports/sessions/2026-10-29-0332*
log has two ready checks motion5.60/4.14, eight text boxes, followed by
reading event:5lines, OCR6.747s, capture0.00030s. Accuracy and page Q&A not
yet assessed.45reading tests passed;84conversation tests passed;full423suite
passed before latest conversation/OCR edits. No feature recording active.
Next: start local server with scripts/run_llama_server.sh, then unrecorded
`python3 -m argus run --demo --windowed --demo-source left --demo-feature reading`
with DISPLAY=:0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Verify OCR accuracy
and page follow-up; test spoken conversation turn-taking next. Do not resume
calibration automatically. Metric navigation/incoming danger warnings still
gated (no accepted calibration); wide-camera SLAM still unimplemented.
Recordings:reports/feature-recordings/2026-10-29/. Design examples and actual
text conversation before/after evidence:reports/interface-redesign/.
Working tree has extensive uncommitted session changes; preserve them.

OCR hold-still stall on live session032832: first frame ready, then26checks
motion7.3–30 despite sharpness1299–4176 and brightness74–115 with5–10boxes.
Motion was comparing frames across~1.1–1.5s OCR detector runs. Changed coach
to sample frame pairs120ms apart BEFORE detection, preserving all quality
gates and two-ready requirement. Current framing cue now updates main OCR
status; log adds motion_sample_s. Regression covers stable short pairs with
brightness change between detector cycles. Reopening unrecorded OCR view;
new capture success still needs live evidence, do not claim it fixed by tests.

Conversation improvement request: server was offline, launched existing
`scripts/run_llama_server.sh` with bounded BLAS/OMP threads (session30263;
log reports/interface-redesign/conversation-server.log). Live production
text checks via new scripts/check_conversation.py, before/after JSONs in
same report directory. Two OCR misroutes fixed for conceptual book/title
questions; persona now direct concrete advice without butler/service filler;
router budget128tokens. Same10case rerun: chat/model routes1.77–2.33sec,
name memory and follow-up context worked; visual query routed describe and
unsupported app action denied. These are text-only, no voice/tool acceptance.
84dialogue/companion tests passed including6new conceptual routing cases.
Restarting unrecorded demo in conversation mode to load fixes. Server should
remain running for voice tests; do not claim microphone flow verified.

User requested operational UI, not promotional demo copy. Removed taglines,
suggested spoken commands, demonstration numbering and default keyboard
hints. Plain feature names, actual OCR/status/requests remain; neutral empty
states. Diagnostics still toggled by Tab. Focused39tests pass. Restarted
unrecorded reading preview (session23628). Previous runtime reported local
reasoning server:8080 connection refused; do not imply conversation is healthy.

User authorized native recording-interface redesign. Implemented
`argus/recording_layout.py` and integrated DemoView: charcoal/cyan 1080p
composition, larger camera, readable masked OCR/transcript, honest activity
and recording states, Tab diagnostics, `--demo-feature` CLI selection.
Recorder remains 960×540; safety overlay stays above diagnostics and existing
face filtering/calibration gates remain. Offline illustrative screenshots
and isolated renderer benchmark are in `reports/interface-redesign/`
(median20.2ms,p9522.0ms; not a full-stack performance acceptance).
Previous feature recording is stopped. Do not resume recordings/calibration
automatically. Live unrecorded reading UI preview is now running with
`--demo --windowed --demo-source left --demo-feature reading` (session43212).
All three cameras discovered. Final suite:423passed in34.73s with bounded
BLAS/OMP threads and host GPU-library access. No new recording started.

Conversation05 take01 finalized30MB; log shows brief greeting/how-are-you
exchange,not broad multi-turn test. User asked next; started privacy06 take01
wide POV with same feature-recording folder/log names. Runtime owns cameras.
User should show consenting face, move slowly/turn slightly, ask scene question,
then shutdown to finalize. Inspect video for missed faces/lag/false blur;
display blur is not proof of model-input privacy. Incoming metric warnings and
wideSLAM still blocked/unimplemented; don't claim complete feature coverage.

User deferred further reading work ("later lets move on"). Reading take02
finalized67MB; actual OCR/spoken reading with errors, no page question tested.
Started next separate Gemma conversation recording:05-conversation-take-01
.mp4/.jsonl in feature-recordings/2026-10-29,wide POV. Runtime owns cameras.
Test open-ended reasoning/chat plus name/follow-up; shutdown to finalize.
Then privacy clip remains. Don't resume OCR/calibration unless asked; retain
all existing limitations and blocked features in manifest.

User asked video analysis. Sampled take01:25–45s title visible, lower page
cut off while old coach stalls;75–105s page absent. Display face blur overlaps
page around65s; video is not raw OCR input. Don't claim continuous/audio review.
Review report `reports/feature-recordings/2026-10-29/reading-video-review.json`.
Take02 currently running, now logged actual OCR reading after five checks
(ready,not-ready,not-ready,ready,ready); motion~5–7.5,brightness~98–101.
Capture fix progressed; text accuracy/page agent still unassessed, MP4 not
finalized. User should ask page question then shutdown to save. Conversation
and privacy separate clips remain next; privacy display-page overlap merits
inspection before claiming robust face blur.

CURRENT:Reading recording take01 finalized146.193s(video/audio verified),but
coach timed out repeatedly and never reached OCR. User reported inconsistent
hold-still/capture. Fixed raw full-image motion to Gaussian-denoised page ROI;
kept brightness/blur/framing gates and two ready checks. Changed first-ready
cue to "The page is framed. Hold it there" and actual selection acknowledgment
to "Captured. Reading now." Stereo OCR uses owned approved coach frame instead
of recapturing different live view. Numeric reading_frame_check telemetry added.
418 tests pass incl exact-frame/cancellation and noise/background regressions.
Take02 prepared at04-reading-page-agent-take-02.mp4/.jsonl in existing recording
folder. Restart user trial with left POV+OCR preview, then page-agent question.
Do not mark reading successful before observed OCR/page question. Seat clip01
already saved214.274s; conversation/privacy still pending. Calibration/wideSLAM
limitations unchanged. Retain all failed takes and no overwrites.

Recording progress:Seat take01 finalized,214.274s,H264 960x540+AAC streams
verified with ffprobe. Seat directions observed (left/ahead), intermittent
target loss/search; no clean acceptance/walking claim. User asked next; started
04-reading-page-agent-take-01.mp4/.jsonl using left POV+local OCR preview.
Runtime now owns cameras. Ask "Argus, read this", then page title question,
shutdown to finalize. Next separate clips conversation then privacy. Manifest
updated in reports/feature-recordings/2026-10-29/session-state.json.

ACTIVE OBJECTIVE CHANGED:User wants separate tests/recordings of stereo seat
finding/navigation, incoming alerts, wide SLAM, OCR+page agent, Gemma conversation,
privacy face blur. Latest sector calibration rejected(20views,stereo2.299,
heldout2.101/8.098px), no activation. Explained seat direction-only possible;
metric warnings/walking blocked; wide SLAM unimplemented(current VO stereo).
Prepared `docs/FEATURE_RECORDING.md` and manifest under
reports/feature-recordings/2026-10-29/session-state.json. Added --demo-source
left/wide (defaultwide), same display face blur, changes recording POV only.
29 relevant tests+CLI/diff checks pass. First seat take launched with left POV:
01-seat-direction-take-01.mp4/.jsonl in that folder. Runtime owns cameras.
User should remain seated, put empty chair in stereo view, say "Argus, where
is the seat?", stop between position changes, shutdown to finalize. Inspect
saved MP4/log before marking success; move next to OCR/conversation/privacy,
keep blocked features honest. Built-in audio records ARGUS,not human mic;
user was told external recording needed for human speech. No feature accepted
or video finalized yet. Do not return to calibration without user's direction.

User authorized sector trial. Added `--detector sector` (default classic),
selected detector used in auto-board detection and both-camera capture, stored
in observation/candidate metadata. SB provides subpixel corners; no classic
refinement applied.46 calibration tests pass, including SB synthetic/blank.
Fresh sector trial command uses23mm/7x7/101mm, output observations suffix
2026-10-29-sector-live. Gates unchanged, live calibration still pending.

Stationary burst DONE and inspected.16pairs,skewmax1.513ms. Raw images under
/tmp/argus-stationary-board-48s2_qfy show noisy/grainy board and horizontal
fold seam; cannot verify flatness from photos. Global motion~7px, so not fully
stationary. Projective-compensated local repeatability classic->sector median:
left.570->.368px,right.311->.216px. Sector15/16 left,16/16 right,classic16/16
left,15/16 right. Report `reports/stationary-board-2026-10-29.json`.
Sector may improve corner localization but repeatability is not ground-truth
accuracy/full calibration acceptance. No runtime detector switch yet. Next
consider opt-in sector detector and improved target lighting/flatness, not
another identical full capture. Preview process may still own cameras until
Q/Esc. No activation.

Stationary diagnostic now launched: `scripts/check_stationary_board.py --out
reports/stationary-board-2026-10-29.json`. User must click preview/press SPACE
with board held still. Captures16 stereo pairs~5s, keeps raw PNGs in unique
/tmp/argus-stationary-board-* for local inspection, numeric report in repo.
Compares classic/sector detectors, planar-grid residual and inter-frame local
repeatability after projective motion compensation. No activation or camera
settings changes. Synthetic identical-frame smoke passed both detectors with
<.001px repeatability. Wait for user burst result; script owns cameras until
Q/Esc. Inspect saved images before attributing physical cause.

MOST RECENT:User completed robust25-view session,7/9 coverage,tilt.175/.427,
skews median.56/max3.55ms. Saved `calibration-observations-2026-10-29-robust-live.npz`.
Fits stable,baseline98.31mm,but mono1.261/1.316,stereo1.411,heldout.570/1.953px
rejected. Offline replay agrees. Shared residual correlation.907; lens variants
barely help, released-board geometry improves several held-out views but not
view20 (>2px). Excluding20 only as sensitivity experiment still leaves mono
1.217/1.221,stereo1.337; no pruning acceptance or original-data modification.
Reports named calibration-replay,target-analysis,residual-pattern,outlier-sensitivity
with2026-10-29-robust-live suffix. Likely shared target/localization issue,
but cannot prove flex/blur without raw images. User did sufficient tilt;
do not blame mounts or ask for another identical full session. Next optional
diagnostic stationary well-lit burst/corner repeatability or verified flat rigid
printed target. Cameras free,no activation. Offline analyses were extremely slow
with unconstrained CPU threads; rerun with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
and cv2.setNumThreads(1). Only own identified slow analysis jobs were stopped.

NOW IMPLEMENTED: optimized3-start intrinsic fitting (lowest mono RMS, no tape
tuning), similar-residual-start focal agreement gate10%. Explicit perspective
tilt guidance via opposite-edge ratios; translation/scale/flat rotation do not
count. Need both cue ranges>=.08 at auto-completion and final acceptance. This
heuristic supplements original quality gates, which are unchanged.413 tests
pass, including initialization/stability and tilt regressions. Replay of latest
old capture:99.396mm baseline,mono.677/.636,stereo1.089,held-out48.19/214.69px;
still rejected. `reports/calibration-replay-2026-10-29-robust.json`.
Next live session should use23mm,7x7,baseline101 and stronger top/bottom tilts;
keep mounts fixed. No accepted calibration or walking readiness.

LATEST: User corrected square size to23mm, confirmed four squares92mm in both
directions. Keep7x7 inner corners; baseline estimate101mm unchanged. Saved
corrected copy of old observations as `calibration-observations-2026-10-29-23mm.npz`.
Fresh23mm capture saved `calibration-observations-2026-10-29-23mm-live.npz`.
Its default fit exploded: baseline2668mm,leftfx7037/fy4877,stereo5.374px.
Replayed and analyzed:3 starting guesses600/800/1000px (fully optimized, not
fixed focal priors) all recover99.396mm baseline,left671/682,right734/746,
stereo1.089px,mono.677/.636px. This demonstrates initialization sensitivity.
No production fitting change yet; no active calibration. Evidence:
`reports/calibration-initialization-2026-10-29-23mm-live.json`, plus replay and
target-analysis with same suffix. Stable solution still fails gates.
Camera pairing skew1.14–3.33ms; joint ordering unchanged on replay. Pose
estimates from earlier intrinsics indicate limited pitch variation, not a
measured ground-truth diagnosis. Next: improve intrinsic initialization and
stability checking, improve explicit out-of-plane tilt capture guidance, and
continue target/localization diagnosis. Do not blame or adjust a specific mount
based on the pathological fit. User asked analysis, not a fresh capture.

New latest: fixed-code live capture (15 views/9 cells) rejected with stereo
RMS .988, mono .818/.835, held-out .926/2.971 px, baseline98.41mm. Observations
saved `reports/calibration-observations-2026-10-29-fixed.npz`; replay report
`reports/calibration-replay-2026-10-29-fixed.json` matches. User authorized
offline target/model investigation. `scripts/analyze_calibration_target.py`
fits views except5/10/15 then tests those held-out. Lens-model variants barely
help; released-board geometry cuts mono held-out RMS from ~.8–1.0 to .3–.4px.
Shape learned on one camera transfers to other's unseen views (.32–.44px).
Report `reports/calibration-target-analysis-2026-10-29.json`. This supports
target-related systematic mismatch, not proof of measured warp/sole cause.
Learned shape is diagnostic ONLY, not active calibration. Next: ask/check
horizontal and vertical square spans (four squares should be96mm) and board
hinge/flatness before another identical session. Cameras free, no activation.

User authorized calibration code inspection/fixes after repeated failures and
mount adjustment. No accepted calibration. Latest old solve: 15 views/9 cells,
83.8 cm baseline, 979/2373 px focal lengths; rejected, no observation replay
possible for prior runs. Cameras currently free; model server may remain up.

Fixed independently canonicalized 7x7 corner mismatch near 45° using joint
centered/scale-normalized grid symmetry matching. Assumes upright roughly
co-directed cameras (<45° relative roll), rejects ambiguous/high-shape-error
matches. Calibration pairing now uses bounded histories, freshness/skew gate
and advancing BOTH sequences. New per-capture NPZ checkpoints (corners/geometry
only) default to active output path plus unique observations timestamp.
`--observations-out` overrides. `scripts/diagnose_stereo_observations.py FILE
--out REPORT.json` replays without activation/pruning. Full suite 408 passes;
known 101 mm synthetic replay recovered 100.984 mm, held-out .069/.278 px.
Report `reports/calibration-code-check-2026-10-29.json`. No relaxed final gates.

Next: fresh live capture to validate fixes; recheck physical lens spacing after
mount adjustment (previous user estimate 100–102 mm, 101 midpoint used), use
24 mm squares and 7x7 corners, good lighting, real out-of-plane tilts and varied
distance. Do not promise fixes explain all previous failures. Avoid another
unrecorded capture: preserve new observations even on rejection. Metric speech
and routes remain gated.

## Latest continuation — 2026-10-28

Calibration inputs received: user says chessboard (assumed 8x8 squares),
square 24 mm, lens baseline 100–102 mm confirmed cm conversion; use 101 mm
midpoint. First --coach 7x7 trial ended 2/15. User complained repeated "good,
hold still" never captured. Fixed hidden same-cell veto before nine views,
made readiness agree with coach (size/diversity/steadiness/cooldown), check
both-camera steadiness and reset when board lost. SPACE/c honor readiness.
No loosened final gate: >=15 diverse views, >=70% coverage, existing RMS,
held-out row and measured-baseline gates. Auto-finish now also requires 70%.
Restart guided capture; calibration remains absent until an accepted solve.

User said OCR is good enough and asked to move on. OCR demo ended; stationary
profile baseline/ocr/combined (20 s each) completed and released cameras.
Baseline clean, OCR 316 ms gap, combined 389 ms gap / p95 103.9 ms; T1 still
partial. Prior long outages show ~203/204 ticks without a stereo pair; they
did not reproduce in this short profile. Do not call them fixed. Evidence:
`reports/reliability-stereo-ocr-2026-10-28-summary.json`. Asked for flat
checkerboard inner-corner grid, square mm and stereo lens-centre baseline mm
to prepare calibration. Do not invent these measurements or enable walking.

User reported inaccurate voice reading and asked to see OCR while testing.
Use `DISPLAY=:0 python3 -m argus run --demo --windowed --ocr-preview`.
Separate local window shows live reading camera, last captured source and
masked actual voice-triggered OCR text/status; no extra OCR pipeline. Raw
diagnostic camera/capture views do not enter recorder or Gemma. Captured
source precedes optional grounder crop/enhancement, so no final OCR boxes are
drawn in misleading coordinates. Camera ownership stays in the runtime.

Three live OCR camera comparisons saved; user says stereo worked much better.
Keep left source. Stereo returned more characters and was faster; exact CER
favored wide but is confounded by title-only reference vs full-frame text and
unverified reference spelling. Do not claim quantitative accuracy superiority.
No 12 MP trial. Next: matched title crops and verified transcription if scoring
accuracy. Interactive comparison process may still own cameras; check before
starting another rig. Report: `reports/ocr-camera-comparison-2026-10-28.json`.

User requested focus untouched, stereo OCR and wide SLAM. OCR routing is built:
`camera.reading_source: left` in repo and live /opt YAML. Coaching and capture
share the live left reader (960x600), without reopening cameras or consuming
stereo pairs. Wide retains optional 12 MP stills. 393 tests pass; live capture
then stereo-pair smoke passed. Real-page accuracy still pending.

Wide metric SLAM is NOT implemented. It needs wide intrinsics/distortion,
stereo-to-wide calibration and validated matched geometry. Do not substitute
wide pixels into stereo intrinsics or feed unscaled monocular poses to metric
mapping. Current VO stays stereo and gated on calibration. Next: seated OCR
trial, investigate late camera outages, then calibration/tracker work.

Human 600.2 s soak FAILED: eight depth outages, three ~20.46 s late in run
with camera reopen logs; cause not established. Model stayed healthy. Summary:
`reports/soak-human-2026-10-28-continued-summary.json`. Cameras passed doctor
at 59.8/59.9/30.0 fps, all USB3. Physical bindings left 2-1.2, right 2-1.1,
wide 2-1.4. Wide enumerated as /dev/video1 in latest smoke: keep discovery.
The model server was started; verify health before reuse. Soak and camera
smoke ended. A fresh `run --demo --record reports/sessions/stereo-ocr-2026-10-28.mp4`
was then launched for the seated reading trial; check runtime ownership before
opening cameras again. No new walking/metric acceptance.

Last updated: 2026-10-21 (Codex companion/local-mapping build; see newest git commit).
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

New conversational commands and local mapping details: `docs/COMPANION.md`.
Use the existing “Argus” wake name. Session memory/timers and background queries
are built; no new assets or cloud services. Walking routes and acoustic
barge-in default **off** in both config copies. No new device acceptance yet.

```bash
cd ~/Argus/Argus
scripts/run_llama_server.sh &                         # Gemma 4 E2B QAT on :8080 (~10 s)
python3 -m argus doctor                               # rig check (cameras, USB3, fps, skew, RAM, OC, llama, Piper)
DISPLAY=:0 python3 -m argus run --demo --record reports/sessions/demo.mp4
#   wake: "Argus, what's in front of me?" | "Argus" -> "Yes?" -> request
#   close: Esc twice within 2 s, or "Argus, shut down" (saves recording + log)
DISPLAY=:0 python3 -m argus run --dashboard --no-mic --ask-file /tmp/ask.txt   # typed questions
python3 -m pytest -q tests/                            # 386 tests; all must pass
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

**Latest user steering:** Jarvis-style conversation plus object detection,
SLAM/navigation/assistance; build software now, physical testing later.
Companion build: `reports/companion-build-2026-10-21.json` (386 synthetic tests),
`docs/COMPANION.md` (commands/limits). New background listening, OCR visual privacy,
sparse keyframe relocalisation and map integration require a fresh full-stack
soak. Do not reuse older performance numbers as evidence for this build.
 Do not
repeat the seated-soak ask until Fardeen resumes testing. New code is synthetic
verified only; the previous T1 timing evidence predates it. Read
`docs/ACCEPTANCE.md` for the built headless service, acceptance scorers, guided
protocols/frame collection, optional Bangla evaluator and VLM variant support.
Service is not installed/enabled. No assets/configuration/pins changed. Synthetic
build report: `reports/build-2026-10-21.json`. All live DoDs remain pending.


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

10. **Headless service:** code/unit/installer built. Later install using
    `python3 scripts/install_user_service.py`, then deliberately test start,
    voice shutdown, crash recovery, logs and boot/USB audio. It does not enable
    user lingering or use sudo. No service is currently enabled by this session.

## 6. Where things live

Additional modules: `companion`, `tasks`, `reminders` (bounded session assistant);
`object_inventory`, `object_search` (local detections); `keyframes`, `local_map`,
`assistance` (relocalisation, map/planner, guarded route skills);
`document_context`, `visual_privacy`, `voice_policy` (boundaries/policy).

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

- **Changed:** `de142de` (`fix(memory): exclude optional torch conversion imports
  from inference`, pushed to origin/main): production import budget, explicit
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


### Session log — 2026-10-21, build now / test later

- **Changes (see latest conventional commit):** seat camera ownership,
  occupied-seat/lost-target rules, TRT serialization/output ownership, robust
  reading/still options, headless supervisor and user unit, acceptance/CER
  scorers, guided rehearsal/frame collection, optional Tesseract Bangla adapter,
  privacy-masked bounded VLM benchmarks with custom model variants and pins.
- **Verification:** 343 synthetic tests; compile checks and CLI help smoke
  checks. `reports/build-2026-10-21.json`. No new camera/audio/GPU/performance
  measurements. The prior 600-second timing result must be repeated on this
  version before inferring equivalent performance. All live acceptance pending.
- **Resume:** read `docs/ACCEPTANCE.md`; review build changes, then T1 real
  workload/fault checks, focus/calibration, reading, walk, seat and wake trials.
  New best-of-N still mode remains opt-in; Bangla evaluator is not the live
  reading backend. Model downloads/switches and PaddleOCR-VL remain pending
  evaluation; optional wake training remains off-device.
- **Exact next [HUMAN] ask (only when testing resumes):** “Stay seated with
  USB headset on, rig pointed into the room away from the monitor, and a printed
  page ready. I will run `DISPLAY=:0 python3 scripts/soak_fast_loop.py --seconds
  600 --out reports/soak-2026-10-21.json`. It gives spoken/displayed prompts
  about every 45 seconds and ends DONE/FAILED. For ten minutes ask room questions
  and try read this, stop, repeat and next page. Then report missed speech,
  failed reads and interruptions.” No human action is requested now.


### Session log — 2026-10-21, conversational companion and sparse local mapping

- **Changes (see latest conventional commit):** Jarvis-style session conversation,
  preferences/name/recap, time-limited follow-ups, single async task and cancelled
  reply suppression; timers; on-demand essential-object inventory and stationary
  search; truthful find phrasing/history; sparse keyframe mapping/geometric
  relocalisation; expiring occupancy grid/inflated A*, named places and guarded
  route execution; default-off acoustic control barge-in. OCR text redaction
  now precedes image model calls; log/model text masking is centralized. Invalid
  depth is UNKNOWN, stale wide-camera views are unavailable.
- **Verification:** 386 tests pass; compile/config/command checks.
  `reports/companion-build-2026-10-21.json`. No new GPU/RAM/latency/OC/live-audio
  evidence. `docs/COMPANION.md` contains commands, interfaces and explicit limits.
- **Resume:** perform deferred device checks on this build. In particular repeat
  T1 with background voice/STT, visual-privacy OCR, inventory, calibrated VO
  relocalisation/map worker, reading and recording. Test tracking loss/recovery,
  changed obstacles/unknown map space, cancellation during model timeouts,
  timers under hazard preemption and echo before enabling optional barge-in.
  Keep `navigation.guided_routes_enabled` false until calibrated walking
  acceptance; this is local sparse mapping, not a global pose-graph/GPS system.
- **Exact next [HUMAN] ask (when testing resumes):** same seated ten-minute
  headset/page soak command in the preceding session block; no walking. After
  that request measured board square size (four squares / four), lens baseline
  in mm and confirmation of flat board for the guided calibration. No new
  physical action or sudo operation is requested in this build session.

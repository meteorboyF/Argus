# ARGUS HCI Hardware Engineering: Obstacles, Decisions, and Evidence

## Purpose

ARGUS is an assistive wearable, not only an AI demonstration. A technically
correct result can still fail its user if it arrives late, interrupts hazard
detection, describes the wrong direction, exposes a bystander's identity, or
depends on fragile calibration. This document records the obstacles encountered
while integrating the physical prototype and explains the decisions taken in
response.

The record distinguishes **implemented and verified**, **temporary workaround**,
and **not yet validated**. Measurements come from the Jetson prototype unless
otherwise stated. The detailed chronological evidence remains in
[`AGENT_LOG.md`](AGENT_LOG.md).

## Human-centred constraints

The following requirements guided engineering decisions:

1. **Safety cannot wait for AI.** Obstacle and drop detection must continue
   while speech recognition or scene description is busy.
2. **Late warnings are unsafe.** A warning must pre-empt ordinary speech and
   stale warnings must be discarded.
3. **Privacy is a precondition.** The vision-language model must never receive
   an unblurred frame.
4. **Uncertain distance must not sound certain.** Metric guidance is disabled
   until calibration is physically verified.
5. **The device must be understandable during development.** Operators need a
   labeled live view, health checks, and measurable timing rather than silent
   failures.
6. **Wearable resources are limited.** Decisions must respect 8 GB unified
   memory, battery, thermals, USB bandwidth, and physical rigidity.

## Decision summary

| Obstacle | Human impact | Decision | State |
|---|---|---|---|
| One heavy pipeline could block hazard sensing | Missed warning while the user approaches danger | Separate a non-ML fast safety loop from the event-driven AI loop | Implemented; independence observed under a >120 s AI stall |
| Speech playback blocked processing | Safety frames were skipped while a phrase played | Asynchronous priority speech; DANGER pre-empts normal speech | Implemented; unit tested, physical pre-emption test pending |
| Stereo frames can be temporally mismatched | Plausible but wrong depth | Reject pairs above 12 ms skew and report degraded loop rate | Implemented; 59/60 pairs within limit in a sustained sample |
| Adjustable camera mounts can move | Calibration silently becomes invalid | Lock mounts, monitor drift, recalibrate after movement | Partly implemented; mechanical and metric validation pending |
| No printed checkerboard available | Calibration blocked | Render a physically scaled checkerboard on the monitor | Implemented; calibration capture still pending |
| Mounted cameras produced sideways images | Incorrect scene interpretation and stereo geometry | Per-camera 90° rotation/flip normalization at capture | Implemented and visually verified |
| Camera drivers expose product IDs, not sensor names | Cameras could be assigned to the wrong role | Discover `B0495` stereo pair and `B0459` wide camera | Implemented and verified with all three cameras |
| All cameras share a USB hub | Dropped or stalled video could degrade safety | Verify 5 Gbit/s links and simultaneous capture before redesigning wiring | Verified for initial prototype; longer soak test remains |
| CPU Gemma response took minutes | Interaction was unusably slow | Disable hidden reasoning; use 256 px input and 48-token answers | Improved cold description to 25.6 s; still not acceptable |
| CUDA offload failed despite reported free memory | GPU acceleration unavailable | Diagnose rather than guess layer counts; upgrade affected R36.4.7 BSP to R36.5 | Root cause confirmed; upgrade staged, not yet verified |
| Three-image Gemma request crashed projector | Temporal/waving interpretation failed | Use one privacy-gated contact sheet until upstream multi-image path is fixed | Temporary workaround |
| Raw headset rejected 16 kHz audio | Wake word/STT input could fail | Use PulseAudio default for resampling, not direct ALSA hardware | Verified with real microphone blocks |
| Piper API had changed | ARGUS produced no spoken output | Consume current Piper `AudioChunk` output directly | Implemented and audibly verified |
| Wide and stereo cameras have different viewpoints | A detected object could receive the wrong depth | Do not claim calibrated distance until cross-camera calibration exists | Known limitation; joint calibration not implemented |

## Detailed rationale

### 1. Safety was separated from conversational intelligence

**Obstacle.** Vision-language inference is variable and slow. A first live
request exceeded two 60-second timeouts. If obstacle sensing shared this path,
the device could become blind while composing an answer.

**Decision.** ARGUS uses two independent loops:

- The fast loop uses synchronized stereo frames, SGBM depth, and geometric
  rules. It contains no neural model and targets approximately 10 Hz.
- The slow loop runs only on demand for speech, privacy filtering, Gemma scene
  understanding, grounding, and spoken response.

**Evidence.** The fast loop continued producing warnings while Gemma was
blocked for more than 120 seconds. This validates architectural isolation, but
does not yet validate hazard distances.

**HCI implication.** Responsiveness for safety is treated separately from the
quality of an AI answer. The conversational feature may degrade without taking
away the user's basic protection.

### 2. Spoken output was changed to respect urgency

**Obstacle.** Speech synthesis and playback originally blocked the calling
thread. A danger warning could wait behind a long scene description, and the
safety loop could skip frames while speaking.

**Decision.** Speech now runs on a background priority queue:

- `DANGER` interrupts current speech and removes queued ordinary speech.
- `WARN` moves ahead of ordinary messages.
- Safety phrases older than 1.5 seconds are dropped instead of being spoken
  after the situation has changed.

**Evidence.** Priority behaviour has automated tests. Piper output was heard
successfully through the connected headset. A complete physical test—start a
long answer, introduce an obstacle, and confirm audible interruption—remains
pending.

**HCI implication.** The auditory channel is managed as a scarce interface.
Urgency, interruption, and staleness matter more than preserving every message.

### 3. Privacy was made a hard system boundary

**Obstacle.** A wearable camera can capture nearby people without their
participation. Sending raw frames to a VLM would create a privacy failure even
if the final description omitted identities.

**Decision.** Every VLM request requires a successful privacy gate. Faces are
blurred before the image crosses the agent boundary; configuration keeps
`require_gate: true`.

**Evidence.** Live preview/description tests passed a gated frame and reported
that one detected face was blurred. Text-region blurring is still unimplemented.

**HCI implication.** Privacy is enforced before inference rather than treated
as a wording rule after inference. The remaining text-blur gap must be disclosed
in any user study.

### 4. Camera identity, orientation, and timing were normalized

**Obstacle.** The Linux camera names did not contain the advertised sensor
names. The two physically rotated stereo modules also delivered sideways
frames. Index-based assignment and untreated rotation could make the preview
confusing and corrupt stereo correspondence.

**Decision.** Hardware discovery uses the reported module IDs:

- `B0495`: the two AR0234 stereo cameras.
- `B0459`: the IMX477P wide camera.

Configurable 0/90/180/270-degree rotation and flips are applied immediately
after capture and consistently in runtime, preview, and calibration. Stereo
pairs over 12 ms apart are rejected.

**Evidence.** All three feeds were opened together. The normalized output was
visually upright. In a 60-pair measurement, 59 pairs were within 12 ms; median
skew was 2.932 ms and p95 was 3.281 ms. The rejected first pair was a 370 ms
startup outlier.

**HCI implication.** The developer sees the same orientation that the system
interprets. Timing checks prefer missing one sample over presenting a precise
but incorrect distance.

### 5. The physical mount became part of the calibration model

**Obstacle.** The stereo modules are installed on adjustable ball joints. A
small bump can change their relative pose while software continues returning
plausible depth. The wide camera is also above and behind the stereo plane,
creating parallax.

**Decision.** Calibration is valid only after the mounts are mechanically
locked. The system includes alignment-drift monitoring, and any mount movement
requires recalibration. Wide-camera detections must not be advertised with
confident metric depth until a wide-to-stereo extrinsic calibration exists.

**Evidence.** Mechanical risk was identified from the prototype assembly.
Metric stereo calibration and tape-measure validation have not yet been
completed; current baseline/focal defaults are placeholders.

**HCI implication.** The interface must communicate uncertainty honestly.
Direction-only guidance is preferable to an incorrect statement such as
"one metre ahead."

### 6. Calibration was adapted to available equipment

**Obstacle.** No printed checkerboard was available, blocking calibration of
the assembled hardware.

**Decision.** `scripts/display_calibration_board.py` renders a checkerboard on
the attached monitor and uses XRandR/EDID physical dimensions to scale its
squares. The headless calibrator can capture poses while the locked rig is
moved in front of the display.

**Evidence.** The monitor reports 1920×1080 over 479×260 mm and can render the
target. This is an enabling method, not completed calibration.

**Acceptance criteria.** Capture at least 15 varied poses, achieve calibration
RMS below approximately 0.6 px and vertical error below approximately 1 px,
then compare estimated distance with a tape measure at several known ranges.

### 7. The three-camera hardware path was validated before adding complexity

**Obstacle.** All three SuperSpeed cameras share one hub, so USB contention
could appear as stalls or dropped frames.

**Decision.** Measure the actual topology and simultaneous capture before
rewiring. Each camera negotiated 5000 Mbit/s behind a 10000 Mbit/s upstream
hub; concurrent capture worked during the initial run.

**Tradeoff.** This avoids unnecessary hardware redesign, but a longer wearable
soak test under full CPU/GPU load is still required.

### 8. Gemma latency was measured and reduced, not hidden

**Obstacle.** The initial CPU-resident Gemma configuration used a 1024-pixel
image, hidden reasoning, a 256-token allowance, and one retry. It failed twice
at 60 seconds and was projected to need more than three minutes.

**Decision.** Keep the requested Gemma 4 E2B model but optimize interaction:

- Disable hidden reasoning because the user needs a short answer, not an
  internal reasoning trace.
- Reduce vision input to a maximum side of 256 pixels.
- Limit output to 48 tokens.
- Remove automatic retry to avoid doubling an already long wait.

**Evidence.** A warm-cache result took 6.41 seconds. The honest cold production
preview took 25.6 seconds: about 12.7 seconds for prompt/image processing and
12.8 seconds for 14 output tokens.

**HCI implication.** The improvement makes testing possible but does not meet
an interactive target. The UI should not imply an immediate answer, and the
system still needs GPU acceleration or a reconsidered slow-loop experience.

### 9. GPU failure was traced to the platform release

**Obstacle.** CUDA offload failed at full, 20-layer, and 4-layer settings with
`NvMapMemAllocInternalTagged error 12`, even when startup reported several
gigabytes free. Guessing smaller layer counts did not address the pattern.

**Decision.** Preserve the CPU fallback, stop blind tuning, and research the
Jetson-specific allocator signature. NVIDIA identifies a known memory issue in
Jetson Linux R36.4.7 and reports it fixed in R36.5. A guarded terminal upgrade
script was added; code was pushed before modifying kernel/firmware packages.

**Evidence.** The device is confirmed on `nvidia-l4t-core 36.4.7`. At the last
inspection it also had only 926 MiB available RAM and all 3.7 GiB swap in use,
so memory pressure remains a separate constraint after the BSP fix.

**State.** The R36.5 upgrade is staged but has not yet been completed and
validated. Success must be determined by a post-upgrade GPU load test, not by
the package version alone.

### 10. Multi-camera temporal interpretation exposed a model-runtime bug

**Obstacle.** A request containing three individual images crashed the Gemma
vision projector with an output-buffer size mismatch. This blocked the planned
"watch a waving motion" experiment.

**Decision.** Combine temporal samples into one privacy-gated contact sheet and
submit it as a single image. Keep the independent three-camera live preview for
human inspection.

**Tradeoff.** A contact sheet preserves coarse motion evidence but is not true
video understanding and may confuse spatial and temporal order. It must be
labeled consistently and treated as a temporary runtime workaround.

### 11. Audio integration required adapting to real device behaviour

**Obstacle.** The USB headset's raw ALSA device accepts 44.1/48 kHz, while
Whisper and the wake-word pipeline use 16 kHz. Direct hardware selection would
fail. Piper's installed API also differed from the older implementation, so no
speech could be produced.

**Decision.** Leave audio devices as the PulseAudio defaults so software
resampling provides valid 16 kHz input. Update Piper playback to consume its
current iterable `AudioChunk` API directly.

**Evidence.** Recording produced 32,000 non-zero samples over two seconds; the
wake-word stream produced consecutive 1,280-sample blocks. The user confirmed
the spoken ARGUS test phrase was clear.

## Decisions deliberately deferred

- **Metric safety thresholds:** unsafe to tune before calibrated depth is
  verified with physical distances.
- **Wide-to-stereo distance fusion:** requires joint extrinsic calibration;
  proportional pixel scaling is insufficient for confident distance claims.
- **True multi-image/video Gemma analysis:** blocked by the current projector
  failure; contact sheets are only a prototype workaround.
- **Full wake-to-answer usability:** requires the R36.5 GPU test, end-of-speech
  detection, and an all-components-resident memory/latency measurement.
- **Text privacy:** face blur works, but text-region blur remains a gap.
- **Field validation:** no obstacle course or study with visually impaired
  participants should begin until calibration, warning pre-emption, and failure
  behaviour have passed controlled safety tests.

## Next validation sequence

1. Complete the guarded R36.5 terminal upgrade and verify kernel, CUDA, all
   three cameras, audio, and the Python test suite.
2. Rebuild `llama.cpp` cleanly and measure full/partial GPU offload with the
   projector initially kept on CPU.
3. Lock and mark the camera joints, run monitor-based stereo calibration, and
   verify multiple tape-measured distances.
4. Perform an obstacle-warning pre-emption test while ordinary speech plays.
5. Measure fast-loop frequency, stereo rejection rate, thermals, power, memory,
   swap, and wake-to-speech latency with the complete system resident.
6. Run controlled failure tests: unplug one camera, cover a lens, move a mount,
   lose the VLM server, and disconnect audio. Confirm the device degrades
   explicitly and never invents precise guidance.
7. Only then design a supervised HCI evaluation covering comprehension,
   workload, trust calibration, warning urgency, privacy expectations, and
   comfort—not just model accuracy.

## Traceability

Primary implementation history: `AGENT_LOG.md` entries #011 and #016–#024.
Current unresolved items: [`KNOWN_GAPS.md`](KNOWN_GAPS.md). Calibration procedure:
[`CALIBRATION.md`](CALIBRATION.md). Hardware assembly constraints:
[`HARDWARE.md`](HARDWARE.md).

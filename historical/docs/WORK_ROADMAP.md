# ARGUS Functional Work Roadmap

## Product outcome

ARGUS should help a blind or low-vision user understand and move through an
unknown environment without pretending uncertain sensor output is reliable.
The intended experience is:

1. Wear the glasses and hear that the system is ready.
2. Receive immediate, concise obstacle and drop warnings without asking.
3. Say “ARGUS” and ask what is nearby or where an object is.
4. Receive privacy-gated scene descriptions and grounded directional guidance.
5. Build a map of an unfamiliar indoor space and receive accessible navigation
   cues while retaining user control.

This roadmap is ordered by dependency and safety risk. A checked item means its
acceptance criteria were measured on the physical Jetson prototype—not merely
that code exists.

## Current verified baseline

- [x] Jetson Linux R36.5.2, CUDA 12.6, Orin GPU tensor execution.
- [x] Gemma 4 E2B multimodal server with full decoder GPU offload.
- [x] Privacy-gated live description: one detected face blurred before Gemma;
  6.90-second response; 8.89 generated tokens/second.
- [x] Three USB 3.0 cameras identified by module name and stable USB topology.
- [x] Simultaneous normalized capture from two B0495 stereo cameras and one
  B0459 wide camera.
- [x] Non-blocking priority speech design and audible Piper output.
- [x] Fast safety loop remains independent during a stalled VLM request.
- [x] Repository regression suite: 34 tests.

## Milestone 1 — Stable hardware and one-command bring-up

- [x] Replace all operator-facing `/dev/videoN` assumptions with camera identity
  and USB-port resolution, including first calibration before a calibration file
  exists. Verified after reconnect renumbered the rig to left/right/wide =
  `/dev/video4`, `/dev/video0`, `/dev/video2`.
- [ ] Add one command that reports power mode, GPU health, three camera identities,
  microphone/speaker, privacy gate, Gemma health, calibration state, RAM/swap,
  and a clear READY/DEGRADED/UNSAFE result.
- [ ] Add controlled recovery for camera disconnect/reconnect and VLM failure.
- [ ] Provide systemd services for headless boot with ordered startup and logs.

Acceptance:

- Reboot and reconnect cameras three times; ARGUS selects the same physical
  left/right/wide roles every time without manual indices.
- Unplug each camera once; the system announces degradation, reconnects, and
  never silently substitutes the wide camera into the stereo pair.

## Milestone 2 — Trustworthy stereo calibration and depth

Calibration is unavoidable with two independently mounted cameras. Their focal
length, distortion, relative rotation, and physical baseline are unknown. Without
those values, disparity cannot become trustworthy metres. The acceptable choices
are: calibrate this pair once after locking the mounts, or replace it with a
factory-calibrated depth camera. The current plan retains the existing hardware
and makes calibration usable.

- [ ] Replace the two-window calibration experiment with one guided command and
  one coherent interface.
- [ ] Bind camera rotations and left/right identity to stable USB ports before
  capture; never rely on V4L2 numbering.
- [ ] Show both views, corner-detection state, pose coverage, captured-view count,
  and actionable instructions without covering the target.
- [ ] Reject poor input early: wrong camera model, mismatched resolution, partial
  board, excessive glare, movement, or unlocked/moved mounts.
- [ ] Solve at runtime resolution and save intrinsics, distortion, stereo
  extrinsics, rectification maps, Q matrix, USB identities, and provenance.
- [ ] Verify recovered depth against tape-measured targets at 0.5, 1, 2, and 3 m.

Acceptance:

- At least 20 well-distributed stereo observations.
- RMS reprojection error below 0.6 px and vertical rectification error below 1 px.
- Median distance within 10% of physical measurements across the useful range.
- Moving either mount causes a visible/aural degraded-calibration warning.

## Milestone 3 — Always-on obstacle and drop warnings

- [ ] Tune SGBM using the verified calibration and realistic indoor surfaces.
- [ ] Define a user-centred hazard region instead of treating every image pixel
  equally; prioritize the walking corridor and head/chest obstacles.
- [ ] Validate obstacle distance, approach direction, drop-off debounce, camera
  skew rejection, stale-frame rejection, and false-positive behavior.
- [ ] Confirm DANGER speech interrupts a long ordinary answer immediately.
- [ ] Add haptic-output support as a redundant warning channel if hardware is
  available; audio remains functional without it.

Acceptance:

- Sustained fast-loop rate at or above 10 Hz under a simultaneous Gemma query.
- Controlled wall/door/chair/person approaches trigger at documented distances.
- Flat floors do not repeatedly announce a drop; real step-down tests trigger.
- Safety warning onset and speech pre-emption latency are measured and logged.

## Milestone 4 — Natural voice interaction

- [ ] Train or integrate a custom “ARGUS” wake-word model; remove “hey Jarvis.”
- [ ] Add voice activity/end-of-speech detection instead of fixed five-second
  recording.
- [ ] Verify microphone capture, Whisper transcription, query cancellation,
  “busy/working” feedback, and Piper output on the wearable audio hardware.
- [ ] Define concise accessible language: urgent warnings, directions, uncertainty,
  failure messages, and repeat/stop commands.

Acceptance:

- Wake-word false accepts/rejects measured in quiet, conversation, and street-like
  noise.
- Median wake-to-transcript and wake-to-first-spoken-answer latency documented.
- “Stop” cancels ordinary speech while safety warnings remain enabled.

## Milestone 5 — Scene questions and object finding

- [x] Single privacy-gated wide-camera description works through Gemma on GPU.
- [ ] Wire and verify YOLO-World grounding for “find X” using the on-device engine
  or an explicitly measured fallback.
- [ ] Add wide-to-stereo extrinsic calibration so a wide-camera detection maps to
  the correct stereo depth pixel; remove proportional-coordinate approximation.
- [ ] Return direction and distance only when confidence and calibrated depth are
  valid; otherwise say what is uncertain.
- [ ] Replace the multi-image crash workaround with a tested temporal/contact-sheet
  protocol and explicit time ordering.
- [ ] Implement text-region privacy blur in addition to face blur.

Acceptance:

- A fixed object set is found across lighting, range, clutter, and left/centre/right
  positions with recorded success rate and latency.
- No unblurred frame reaches Gemma in tests that intentionally fail the privacy
  model.
- Spoken distances pass physical checks; low-confidence detections do not become
  confident instructions.

## Milestone 6 — SLAM and localization

SLAM is not currently implemented. Monocular wide-camera SLAM can estimate motion
and build a map, but absolute scale and robustness improve substantially with an
IMU or stereo/depth input. Before implementation, inventory whether this prototype
has an IMU; if not, choose explicitly between adding one and accepting monocular
limitations.

- [ ] Select and benchmark an ARM64/Jetson-compatible SLAM backend (ORB-SLAM3 or
  OpenVINS-based design) against the B0459 camera, calibration, and available IMU.
- [ ] Calibrate wide-camera intrinsics and, if present, camera-to-IMU timing and
  extrinsics.
- [ ] Build map/save/load/relocalize commands and expose tracking quality.
- [ ] Detect tracking loss and stop navigation guidance rather than dead-reckoning
  confidently.

Acceptance:

- Complete multiple indoor loops with bounded trajectory drift.
- Save a map, restart, and relocalize in the mapped area.
- Rapid turns, blank walls, lighting changes, and partial occlusion produce
  explicit tracking-quality transitions.

## Milestone 7 — Accessible navigation behavior

- [ ] Build a traversability/topological layer from SLAM pose plus calibrated
  obstacle sensing; a raw SLAM point cloud is not a navigation product.
- [ ] Support user-defined landmarks and destinations, route planning, turn cues,
  off-route recovery, and “where am I?”
- [ ] Keep navigation advisory: never claim a path is safe solely because SLAM
  produced a route; the independent fast safety loop remains authoritative.
- [ ] Design concise egocentric instructions and avoid audio overload.

Acceptance:

- Controlled routes with turns and obstacles complete without silent tracking
  loss or contradictory safety/navigation speech.
- The user can repeat, pause, cancel, and request orientation at any time.
- Navigation stops safely when localization or depth confidence is insufficient.

## Milestone 8 — Headless integration and product validation

- [ ] Run all required components as monitored services without GNOME, Chrome, or
  VS Code consuming deployment memory.
- [ ] Measure steady and peak RAM/swap, GPU/CPU use, power, thermals, camera rate,
  safety latency, query latency, and sustained operation.
- [ ] Add rotating logs and exportable session diagnostics without retaining raw
  bystander imagery by default.
- [ ] Perform fault injection: camera loss, microphone loss, speaker loss, VLM
  crash, low memory, overheating, calibration drift, and power interruption.
- [ ] Complete supervised HCI studies covering comfort, comprehension, workload,
  privacy expectations, trust calibration, warning urgency, and failure recovery.

Acceptance:

- Two-hour wearable soak test without thermal shutdown, runaway swap, deadlock,
  silent sensor loss, or safety-loop starvation.
- Documented recovery behavior for every critical component.
- No unsupervised mobility claims or deployment with blind participants until
  controlled safety review and ethics/accessibility procedures are complete.

## Immediate execution order

1. Finish Milestone 1 camera identity/bring-up reliability.
2. Deliver the single-command calibration workflow in Milestone 2.
3. Capture and verify real calibration data with the user moving the locked rig.
4. Tune and validate safety warnings before adding SLAM complexity.
5. Complete voice and grounded Q&A.
6. Inventory/add IMU and implement SLAM.
7. Build navigation behavior and conduct supervised HCI validation.

## Definition of “working MVP”

The first honest MVP is not full autonomous navigation. It is reached when the
wearable boots headlessly, identifies its sensors, maintains calibrated 10 Hz
hazard warnings, accepts “ARGUS” voice questions, privacy-gates images, describes
the scene or locates an object, speaks concise guidance, exposes degraded states,
and survives controlled failure tests. SLAM navigation is the next product
increment after this safety foundation passes.

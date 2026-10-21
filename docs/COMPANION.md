# ARGUS conversational companion and local mapping

Built 2026-10-21. Device acceptance is pending. These commands use the existing
local Gemma, Whisper, Piper and TensorRT assets; no new model or cloud service
is required. Start ARGUS using the commands in AGENT_HANDOFF.md. Typed demo
requests and the name-based voice listener share the same assistant tools.

## Talk naturally

Say “Argus” followed by a request. After a completed reply there is a bounded
20-second follow-up window without repeating the name. The name-based listener
continues listening while a background task is computing. One slow task runs at
a time; additional tasks receive an honest busy reply instead of accumulating.
Stop/cancel bypasses the query lock and cancels reading/search/route work. An
in-flight HTTP request may finish or time out, but its cancelled reply is not
spoken. This is cancellation of effects, not a claim that model computation
stops immediately. Natural conversation uses the existing local model and a
short, masked history. Malformed router output asks for clarification rather
than opening the camera for an unrelated request.

Examples:

- “Hello.” “What can you do?” “Tell me a joke.”
- “Call me Fardeen.” “What is my name?”
- “Be brief.” “More detail.” “What did we talk about?”
- “What are you doing?” “Where did you last see it?”
- “Find my keys.” “Is it still there?” “Watch my keys.”
- “What objects can you detect?” “Read this.” “What does the page say?”
- “Set a timer for five minutes.” “Remind me in ten minutes to check my tea.”
- “Cancel all timers.” “Forget everything.” “Argus, stop.”

Session memory is bounded: four completed conversation turns, up to 32 recent
object observations, 32 named map places, and 16 timers. Object statements say
which past view they came from and expire; “is it still there?” re-runs detection.
Names and preferences are session-only. Nothing here persists after shutdown.
“Forget everything” clears session history, observations, preferences, map/places
and timers; it explicitly does not erase existing study logs. User text is masked
before logs, display and model requests. Document questions use bounded masked
excerpts when necessary, with an explicit excerpt label.

Timers accept whole seconds/minutes/hours from 1 second to 1 day. Spelled
numbers supported: one through twelve, fifteen, twenty, thirty, forty, sixty.
They do not survive restart and do not call/message anyone. Unsupported duration
requests cannot fall through to a model claiming a timer was created.

## Object assistance

The on-demand inventory reuses the existing YOLO11n TensorRT engine and detects
a selected set of common COCO objects. It groups counts/directions, retains
observations only briefly, and reports misses as misses in the current view.
The safety detector's original moving-object class filter remains unchanged.
Open-vocabulary “find” still uses the installed YOLO-World embeddings; unknown
labels fail closed without loading Torch. Search/watch requires two consecutive
positive detections within a bounded stationary scan. It can say “turn slowly”
while searching; it does not tell the wearer to walk toward an unverified object.

Find results are phrased from the detector result in code, so a language model
cannot manufacture a detection or distance. Wide-camera results have direction
only. There is still no verified wide-to-stereo distance projection. A missing
or older-than-500-ms wide frame is unavailable. Visual model inputs pass the
face gate, then local OCR sensitive-box redaction. If either stage fails, the
image request is refused. Image privacy still needs an adversarial live set;
OCR cannot guarantee recognition of every sensitive number.

## Local sparse mapping and navigation

The existing stereo VO now retains a bounded sparse map of up to 24 keyframes
and supports ORB descriptor/PnP-RANSAC relocalisation with uniqueness, inlier,
positive-depth and reprojection gates. It never resumes a metric trajectory
across tracking loss merely because features reappear. Source timestamp gaps,
stale poses, and failed geometric recovery invalidate the pose. The original
synthetic two-depth-plane motion test remains; additional tests recover a known
pose and reject unrelated geometry. This is sparse local mapping/relocalisation;
it is not a globally optimised pose graph, a surveyed building map, or proof
against perceptual aliasing and long-term drift.

An independent latest-only worker integrates matched calibrated depth/pose
snapshots into a bounded 96×96 grid at 0.25 m resolution. Cells expire after
30 seconds. A* treats unknown/expired cells as blocked and inflates blocked
space by a 0.4 m body radius. The map is an obstacle observation map, not proof
of floor support or accessibility. It stores no camera images. Local coordinates
start at session origin; there is no street address, GPS position or assumed
room label.

Commands:

- “Remember this place as desk.” “List places.”
- “Where am I?” “Tracking status.”
- “Guide me to desk.” “Return to desk.”
- “Guide me to a chair” uses seat finding; marked destinations use the local map.

Marking/navigation requires calibration, green calibration health, current valid
tracking and sufficient pose quality. A plan can be described while walking
instructions remain disabled. `navigation.guided_routes_enabled` defaults to
false in both config copies. Enable it only after the calibrated navigation
acceptance checks below. The implemented route follower additionally requires
current depth, a matching map frame, current valid ground/hazard observations,
no drop, CLEAR safety state and a clear local corridor. Losing any requirement
ends guidance with a stop instruction. Slow tasks never suppress the safety
loop. The map/planner remains experimental until measured on the actual rig.

## Listening during ARGUS speech

`speech.barge_in_enabled` defaults to false in both config copies. Optional
headset barge-in accepts only explicit wake-prefixed stop/pause commands while
TTS is active. Ordinary room speech cannot become a task during TTS. The text
policy is unit-tested; acoustic echo, real microphone latency and false controls
are not measured. Do not describe it as accepted full-duplex conversation.
Without this opt-in, speech controls are heard during gaps in ARGUS speech,
and typed controls remain available immediately. Hazard preemption is independent.

## Tests to perform later

Repeat the T1 combined soak on this version: background listening/STT, OCR privacy
redaction, calibrated VO relocalisation/map worker, inventory/search, reading and
recording all need measured latency/RSS/MemAvailable/GR3D/OC evidence. The prior
soak does not validate these changes. Then complete focus/reading acceptance,
gated calibration, annotated supervised walks, occupied-seat trials and wake
recall/ambient tests. Include deliberate tracking loss, return to a keyframe,
repeated textures, moving people, changed obstacles, narrow passages, blank
walls, disconnected cameras, model timeout during cancellation, long documents,
TTS echo/barge-in, reminder delivery during hazard speech, and shutdown cleanup.
Use `docs/ACCEPTANCE.md` and the existing report tools. No walking acceptance,
full-duplex claim or global SLAM accuracy claim is made by synthetic tests.

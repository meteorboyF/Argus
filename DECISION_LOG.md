# ARGUS engineering decision and hurdle log

This is a required HCI research artifact. Each entry records the real constraint,
its user/system impact, alternatives, resolution, and continuing consequence.
It must be updated with every non-trivial feature decision or setback.

## 2026-05 to 2026-06 — LocateAnything-3B was infeasible on Orin Nano

**Hurdle / problem.** The proposed LocateAnything-3B grounder shipped BF16-only
weights, had no deployable GGUF/ONNX/TensorRT path, exceeded the practical 8 GB
resident budget, and depended on kernels targeting newer Hopper/Blackwell-class
GPUs rather than Orin's Ampere GPU.

**Impact.** The original grounder could not coexist with the reasoning model and
safety workload, so the promised named-object feature had no credible device
deployment path.

**Options considered.** Attempt unsupported quantization/export; swap models in
and out; move grounding to cloud; use NanoOWL; use YOLO-World.

**Resolution.** Pivot the grounding role to YOLO-World, with TensorRT as the
required production backend. This preserved on-device open-vocabulary intent
without model swapping or cloud dependence.

**Lesson / consequence.** Vendor/model-card deployment constraints must be
verified before architecture claims. YOLO-World is still partial until its real
TensorRT vocabulary contract and latency are proven on this engine.

## Pre-audit, exact date not recorded — unfinished `/etc/fstab` entry caused emergency boot

**Hurdle / problem.** An unfinished placeholder mount was left in `/etc/fstab`.
The optional target could not mount during boot, sending the Jetson into emergency
mode.

**Impact.** The wearable compute unit became unavailable and required recovery;
a storage convenience change created device-level operational risk.

**Options considered.** Remove the optional mount, make it a managed service, or
retain it with boot-safe mount options and validation.

**Resolution.** Recover the boot configuration and establish the rule that every
optional `/etc/fstab` entry uses `nofail` and is validated with `mount -a` before
reboot. Placeholders must never be committed to the active system file.

**Lesson / consequence.** Boot configuration is safety-critical. Future system
changes require backup, exact-target review, validation, and recovery notes.

## 2026-08-14 — L4T R36.5.2 recovered Gemma CUDA execution

**Hurdle / problem.** On R36.4.7, llama.cpp CUDA allocations failed with Jetson
NVMAP errors even at small offload counts despite apparently free shared memory.
CPU multimodal inference worked but was too slow for useful interaction.

**Impact.** The selected on-device reasoning model could not meet interaction
latency and repeated blind tuning risked instability without addressing the BSP.

**Options considered.** Remain CPU-only; incrementally offload layers; alter mmap
and unified-memory flags; replace Gemma; downgrade/reflash; upgrade the affected
Jetson Linux release.

**Resolution.** Back up device state, move from R36.4.7 to R36.5.2, rebuild
llama.cpp at commit `ef8268feee28ae943958049bf3bbab4bda99c0ea` for CUDA arch
87, and verify full decoder offload with a privacy-gated image query.

**Lesson / consequence.** Jetson BSP behavior can dominate apparent model-memory
failures. Preserve exact platform/build provenance and verify GR3D activity;
successful loading alone is not a full-stack performance result.

## 2026-08-14 audit — silent CPU fallbacks violated the GPU mandate

**Hurdle / problem.** Documentation described GPU depth and TensorRT grounding,
but configuration defaulted to CPU SGBM and runtime grounding always loaded the
Ultralytics PyTorch model. Missing engines silently degraded instead of failing.

**Impact.** Performance claims were invalid, GPU/CPU contention was hidden, and
the safety loop could run too slowly while presenting itself as operational.

**Options considered.** Keep transparent fallback for convenience; expose a
diagnostic-only fallback; or fail all startup whenever GPU backends are absent.

**Resolution.** Production now fails closed. CPU SGBM and PyTorch grounding may
only be explicit diagnostic modes and cannot support a wearable demo.

**Lesson / consequence.** Backend identity is part of correctness. Every model
report must state the loaded backend and show latency plus `tegrastats` GR3D
evidence.

## 2026-08-14 audit — YOLO-World engine may contain a fixed vocabulary

**Hurdle / problem.** The archived export notebook called `set_classes()` with
six names before ONNX export. The resulting TensorRT engine may therefore bake
those classes and cannot yet be assumed to accept an arbitrary user phrase.

**Impact.** Building the object-finding agent on this engine could silently break
the core promise that a user can name any object.

**Options considered.** Fixed documented vocabulary; rebuild per query; expose
prompt embeddings as an engine input; separate/cache the text encoder; select a
different supported open-vocabulary TensorRT implementation.

**Resolution.** Mark the current engine unverified and make production startup
refuse it until its binding/graph contract is inspected and tested. Resolve this
before further grounding integration.

**Lesson / consequence.** “YOLO-World” model identity does not prove dynamic
vocabulary survives export. The deployed graph contract is the authority.

## 2026-08-14 audit — proportional wide-to-stereo fusion was geometrically invalid

**Hurdle / problem.** A detection from the offset, wide-lens IMX477P camera was
mapped into the stereo depth image by proportional pixel scaling. The cameras
have different intrinsics, extrinsics, distortion, and field of view.

**Impact.** ARGUS could speak a confident distance belonging to a different
surface, creating direct navigation risk for a blind user.

**Options considered.** Keep approximate distances with a disclaimer; use only
the wide camera for direction; estimate a homography; perform full calibrated
cross-camera projection.

**Resolution.** Disable object distance and return direction only. Distance stays
absent until wide-camera intrinsics and wide-to-stereo extrinsics are calibrated,
validated across the overlap, and paired with fresh valid depth.

**Lesson / consequence.** Plausible geometry is not safe geometry. Unknown must
remain unknown rather than being converted into fluent but false speech.

## 2026-08-14 audit — documentation overstated implementation

**Hurdle / problem.** The README and design documents presented SLAM, TensorRT
grounding/depth, CRAFT privacy, and passing integration as current capabilities.
The code showed these were absent, diagnostic-only, or broken; even the claimed
36-test pass was a suite that hung after 31 tests.

**Impact.** A fresh agent could build on false assumptions, supervisors could be
shown misleading progress, and unsafe prototype behavior could be mistaken for a
validated assistive function.

**Options considered.** Patch individual contradictions; keep multiple roadmaps
with warnings; or collapse active guidance to a small authoritative set and
archive the rest.

**Resolution.** Preserve old material under `historical/`, mark PDF/DOCX as
DESIGN INTENT, and reduce living guidance to README, ARCHITECTURE, STATUS,
AGENT_HANDOFF, and DECISION_LOG. STATUS is the implementation authority.

**Lesson / consequence.** Documentation is part of the safety boundary. It must
trail verified reality, never lead it, and all three continuation records must be
updated in the same feature commit.

## 2026-08-14 cleanup gate — fail-closed baseline established

**Hurdle / problem.** Prototype-friendly fallbacks allowed uncalibrated depth,
unwired GPU paths, privacy exceptions, and unverified camera fusion to continue
into user-facing answers.

**Impact.** The system could remain available by becoming less truthful—the
wrong tradeoff for safety-critical HCI.

**Options considered.** Warnings in logs, reduced-confidence speech, or explicit
startup/query refusal at each unsafe boundary.

**Resolution.** Enforce refusal in code: production requires GPU backends;
uncalibrated stereo cannot produce hazard speech; privacy exceptions cancel the
query; and object distance is omitted without calibrated projection.

**Lesson / consequence.** Early supervisor demos may expose missing features,
but must never manufacture confidence. Availability is secondary to truthful
degradation.

## 2026-08-14 Feature 1 — measured environment baseline replaced assumption

**Hurdle / problem.** Prior notes mixed observed platform facts with inferred
readiness. CUDA imports, model files, ALSA's internal devices, and a serialized
engine could each produce a reassuring check without proving GPU activity,
correct USB hardware, artifact identity, or an executable production path.

**Impact.** A fresh agent could start integration on the wrong backend or report
the device ready while clocks, calibration, or GPU depth were missing.

**Options considered.** Continue extending the old print-only self-test; use a
shell checklist; or create a structured, non-mutating diagnostic with required
versus advisory checks and a JSON evidence record.

**Resolution.** Add `python3 -m argus baseline`. It verifies user/platform/power,
runs CUDA matrix work while sampling `tegrastats`, checks exact packages and
hashes, inspects the pinned llama.cpp build, enumerates the three named cameras,
USB topology and USB audio, records memory/zram, and checks calibration/engines.
The 2026-08-14 run passed 22 checks and failed three: locked clocks could not be
queried without interactive sudo, stereo calibration was absent, and no GPU
depth engine existed. CUDA reached 99% GR3D and PyCUDA initialized one Orin GPU.

**Lesson / consequence.** Readiness is now machine-readable and fail-closed.
Existence and imports remain evidence, not completion; GPU claims require a real
workload and GR3D measurement.

## 2026-08-14 Feature 1 — PyCUDA source build needed explicit Jetson paths

**Hurdle / problem.** Installing pinned PyCUDA 2024.1.2 initially failed with
`cuda.h: No such file or directory` and warned that `nvcc` was not on PATH, even
though JetPack had both under `/usr/local/cuda`.

**Impact.** The existing TensorRT runner could not allocate CUDA buffers, blocking
both GPU grounding and depth integration.

**Options considered.** Change PyCUDA versions; use an unpinned third-party wheel;
rewrite immediately around another CUDA binding; or compile the pinned source
with explicit Jetson toolkit paths.

**Resolution.** Keep version 2024.1.2 and build on-device with `/usr/local/cuda/bin`
on PATH plus explicit `CUDA_ROOT`, `CUDA_INC_DIR`, and `LIBRARY_PATH`. The wheel
built successfully and reported one device named Orin. Those exports and the
resolved transitive versions are now pinned in setup.

**Lesson / consequence.** JetPack installation does not guarantee Python build
systems discover CUDA. Reproducibility includes compiler/include/library paths,
not only package versions.

## 2026-08-14 Feature 2 — restored open vocabulary with a runtime embedding

**Hurdle / problem.** The inherited ONNX exposed only `images` and produced
`[1,84,8400]` with COCO-80 metadata. Its TensorRT engine therefore had a fixed
COCO vocabulary; the earlier claim that six notebook labels were baked in was
also inaccurate. Calling the artifact YOLO-World did not make it open-vocabulary.

**Impact.** `find_object("keys")` could not honor an arbitrary spoken name, and
building agent behavior on the legacy engine would silently break the product's
central interaction promise.

**Options considered.** Admit a COCO-only product; rebuild an engine for every
query; keep PyTorch YOLO-World in production; or expose CLIP text features as a
TensorRT input while fixing the class count to one.

**Resolution.** Export a two-input graph on the Jetson: image
`[1,3,640,640]`, normalized text embedding `[1,1,512]`, output `[1,5,8400]`.
TensorRT 10.3 built the FP16 engine on-device in 604 seconds. A pinned CPU
ViT-B/32 encoder supplies and caches requested-label vectors. Cached queries
measured about 30 ms; a new label with warm encoder measured 0.46 s. `tegrastats`
showed GR3D activity, and TensorRT/PyTorch score mean absolute error was
3.36e-7. The test scene had no positive detection, so physical accuracy remains
explicitly partial.

**Lesson / consequence.** Engine bindings are the vocabulary contract. Runtime
embeddings preserve the interactive promise without PyTorch detection fallback,
but numerical agreement and latency do not replace positive physical-object
validation.

## 2026-08-14 Feature 2 — text encoding was pinned and kept on CPU

**Hurdle / problem.** Ultralytics would auto-install a moving CLIP Git branch and
download weights into a working-directory-dependent cache.

**Impact.** The same spoken label could not be reproduced from a clean device,
and an implicit GPU text encoder would contend with depth, grounding, and Gemma.

**Options considered.** Accept auto-install; precompute a fixed vocabulary;
move CLIP to GPU; or pin source, dependencies, weight bytes, and run it lazily on
CPU.

**Resolution.** Pin the Ultralytics CLIP fork at commit
`488e81a6711eea7346872b46ea928b367da8889d`, pin its dependencies, store the
audited ViT-B/32 file under `/opt/argus/models/clip`, verify SHA-256
`40d365...50af`, and cache up to 64 label vectors. Cold initialization measured
11.08 s in the integrated query; subsequent new labels were sub-second.

**Lesson / consequence.** Prompt encoding belongs outside the GPU-heavy path,
but it needs warm-up before a demo. Cache behavior and cold-start latency are
part of the user experience and must be reported separately.

## 2026-08-14 Feature 3 — GPU depth used deterministic CUDA, not imaginary RAFT

**Hurdle / problem.** The configuration named a RAFT-Stereo TensorRT backend,
but no RAFT model, ONNX, engine, provenance, or export path existed. The installed
OpenCV Python package also reported no CUDA stereo implementation.

**Impact.** Production correctly refused to start, leaving the always-on safety
loop without depth. Pulling a large unverified network would also threaten the
8 GB co-residency target and weaken the fast loop's auditability.

**Options considered.** Acquire/export RAFT-Stereo; rebuild OpenCV with CUDA
StereoBM; keep CPU SGBM; or implement a small deterministic CUDA block matcher.

**Resolution.** Add a PyCUDA 5x5 SAD matcher compiled on-device for SM 8.7. It
runs at half resolution, searches 64 working-resolution disparities, restores
full-resolution pixel units, applies a uniqueness check, reuses GPU buffers, and
never falls back to CPU in production. A synthetic 12-pixel shift test passed.
On 60 live runs it measured 12.7 ms median, 14.1 ms p95, GR3D peak 95%, and
3,985–4,114 MB total RAM during sampling.

**Lesson / consequence.** GPU acceleration does not require a learned model.
For the safety loop, a small inspectable kernel better matches the architecture
and memory budget. RAFT remains optional until it has pinned provenance and
demonstrates a material accuracy benefit within the same latency/memory budget.

## 2026-08-14 Feature 3 — performance did not imply safe distance

**Hurdle / problem.** Live stereo pairs had 1.03 ms median skew but the initial
pair reached 369 ms, and no physical stereo calibration file exists. Only 6.5%
of pixels passed matching on the unrectified static scene.

**Impact.** Fast GPU disparity could still become confidently false metric depth,
especially during head motion or with the cameras' non-coplanar mounting.

**Options considered.** Loosen the skew gate; speak rough focal/baseline distance;
disable all depth work; or keep diagnostic disparity while rejecting high-skew
pairs and suppressing metric safety speech.

**Resolution.** Preserve the 12 ms pair gate and existing calibration precondition.
The CUDA backend may run and share diagnostic disparity, but the safety evaluator
receives nothing until calibration is present. The feature is marked partial,
not complete, pending 0.5–3 m physical validation.

**Lesson / consequence.** Backend, synchronization, and calibration are separate
acceptance gates. Passing the GPU mandate satisfies only one of them.

## 2026-08-14 Feature 3 — power mode had regressed to 15W

**Hurdle / problem.** The refreshed structured baseline reported current power
mode `15W`, contradicting the earlier MAXN_SUPER observation. `jetson_clocks`
also refuses non-root execution and sudo requires the user's interactive password.

**Impact.** The measured CUDA depth latency is valid GPU evidence but not the
final maximum-performance result required for the wearable. Other model latency
and co-residency measurements would also be misleading if labeled MAXN.

**Options considered.** Reuse the historical MAXN claim; attempt to bypass sudo;
pause all work; or retain the conservative 15W measurement and require the user
to apply power settings before final performance acceptance.

**Resolution.** Record 15W in the depth report and keep baseline readiness red.
The user must run `sudo nvpmodel -m 0 && sudo jetson_clocks`; the baseline and
depth benchmark must then be rerun. No credential was requested or bypassed.

**Lesson / consequence.** Power mode is mutable runtime state, not a platform
constant. Every performance acceptance report must capture it at measurement
time.

## 2026-08-14 Feature 4 — CUDA context failed across the safety thread

**Hurdle / problem.** The CUDA matcher initialized successfully on the main
thread, then failed with `cuMemAlloc failed: invalid device context` when the
safety thread performed its first allocation. The thread died while the slow
query continued and spoke an answer.

**Impact.** A demo could appear successful after silently losing its independent
safety loop—the exact failure the Two-Speed architecture is meant to prevent.

**Options considered.** Construct depth inside the worker; create a context per
frame; ignore the thread exception; or share CUDA's retained primary context and
make startup wait for the first real depth result.

**Resolution.** Both PyCUDA users now retain the primary context. CUDA depth
pushes/pops it on the calling thread. The fast loop catches fatal errors, signals
startup, and `start_fast_loop` refuses to proceed until a valid GPU result exists.
The spoken rerun kept the fast loop alive concurrently.

**Lesson / consequence.** Successful GPU initialization is not runtime proof
when work crosses threads. Readiness must include the first inference on the
actual production thread, and worker failure must propagate.

## 2026-08-14 Feature 4 — explicit locate intent bypassed the tool

**Hurdle / problem.** Gemma answered “I cannot locate the chair” directly rather
than emitting the requested tool JSON, despite a system instruction not to guess.

**Impact.** Natural-language compliance alone could bypass verified grounding
and give an unsupported location answer to a blind user.

**Options considered.** Prompt harder; switch chat templates; accept the answer;
or route unambiguous find/locate/where requests through grounding in deterministic
orchestration code.

**Resolution.** Add a tested locate-intent parser. If Gemma omits `find_object`
for an explicit request, the orchestrator forces that tool and discards the
ungrounded text. “Find the monitor” then produced a positive center result with
distance omitted.

**Lesson / consequence.** Safety-relevant tool policy belongs in code, not only
in a probabilistic prompt. Gemma may decide how to explain evidence, but it may
not decide to skip required evidence.

## 2026-08-14 Feature 4 — PortAudio underruns on the USB Pulse sink

**Hurdle / problem.** Piper synthesis succeeded, but sounddevice playback emitted
repeated ALSA underruns through the default PulseAudio USB sink. High-latency
PortAudio buffering did not resolve them; direct `paplay` was clean.

**Impact.** A generated warning or answer is useless if playback stutters, and
enumerating a device is not evidence that the user receives intelligible audio.

**Options considered.** Keep PortAudio; target raw ALSA while fighting Pulse for
the device; add larger PortAudio buffering; or send raw Piper audio to Pulse and
retain a preemptible child process.

**Resolution.** Default-output TTS now streams float32 audio to `paplay`; explicit
device selections retain the PortAudio fallback. DANGER preemption terminates the
active Pulse process. The clean path completed without underrun messages, and all
six priority/preemption tests pass.

**Lesson / consequence.** Capture/playback enumeration is only a baseline.
End-to-end audio needs backend-specific playback evidence and preemption tests.

## 2026-08-14 Feature 4 — honest demo passed but memory remains over target

**Hurdle / problem.** The complete positive query reached 7,376 MB RAM and used
1,710–3,535 MB swap. Cold grounding took 15.545 s because loading the CPU CLIP
encoder dominates its otherwise ~30 ms cached TensorRT inference.

**Impact.** The supervisor thread is visible and truthful, but peak memory is too
close to the 8 GB ceiling and cold latency is not wearable-quality.

**Options considered.** Hide cold start in the demo; remove arbitrary vocabulary;
move CLIP to GPU; or accept this milestone while scheduling text-encoder and
co-residency hardening.

**Resolution.** Keep arbitrary runtime vocabulary and CPU CLIP for now, record
stage-level latency and memory, and mark integration partial. The current demo
uses MAXN_SUPER, GR3D peaked at 99%, and all false distances remained suppressed.

**Lesson / consequence.** An early end-to-end thread is a diagnostic milestone,
not completion. Cold and cached latency, RAM, swap, and safety degradation must
be reported separately.

## 2026-09-29 Dashboard build — Laya was evaluated and rejected for hazards

**Hurdle / problem.** The user asked whether Laya (a 421M ModernBERT decision
model answering typed questions over text/JSON) could make the "car is coming"
warning faster.

**Impact.** Putting a probabilistic text model in the hazard path would break
the deterministic fast-loop rule and cost about 1 GB of unified memory the
stack cannot spare.

**Options considered.** Laya as hazard gate; Laya as slow-loop intent router;
pure geometry.

**Resolution.** Time-to-collision is geometry: per-zone range history from the
depth map, a linear fit for closing speed, TTC = range / speed. Implemented in
`safety.py` with WARN at 3 s and DANGER at 1.5 s. Laya is parked as a possible
later intent-router experiment; a regex already routes locate requests.

**Lesson / consequence.** A model is not a shortcut for a physics problem the
fast loop already has the numbers for.

## 2026-09-29 Dashboard build — cameras fell back to USB 2 on the same hub port

**Hurdle / problem.** After the user re-plugged the cameras, two of them
enumerated on the High-Speed companion bus (`1-2.2`, `1-2.4`) instead of the
SuperSpeed bus (`2-1.2`, `2-1.4`), delivered 10 fps and stalled for seconds.
Exact USB-path binding then failed to find them.

**Impact.** The runtime refused to start, and a 10 fps stereo camera makes
almost every pair fail the 12 ms skew gate.

**Options considered.** Bind by the physical hub port; bind by serial (the
Arducams expose none); ignore link speed.

**Resolution.** Roles bind to the hub port number and the runtime prints a
warning with the negotiated link speed. Per-camera capture threads record real
arrival times, and a camera whose read blocks now reports 0 fps after 2 s.
The physical fix is still a reseat; the software only makes the fault visible.

**Lesson / consequence.** A USB 3 hub is two buses. Bind roles to what does
not change, and surface link speed instead of letting it hide as skew drops.

## 2026-09-29 Dashboard build — single-plane synthetic scene cannot test odometry

**Hurdle / problem.** The first stereo-odometry test rendered one
fronto-parallel wall and expected a 0.12 m sideways translation; PnP returned
0.08 m with 100% inliers.

**Impact.** A test that passes or fails on a degenerate scene says nothing
about the estimator.

**Resolution.** On a single plane at one depth a sideways translation and a
small yaw produce the same image shift, so the split between them is
arbitrary. The fixture now has two textured layers at 2 m and 3 m, which makes
the motion unique; the test also asserts near-zero yaw. Fixed the fixture's
disparity sign at the same time.

**Lesson / consequence.** Synthetic geometry tests must be well-posed before
their tolerances mean anything.

## 2026-10-02 M0 rig doctor — the deployed config had silently drifted

**Hurdle / problem.** The first `argus doctor` snapshot came back portrait
(600x960). `load_config()` defaults to `/opt/argus/config/argus.yaml`, and that
copy dated from 2026-08-13: `left_rotation: 90`, `right_rotation: 270` and
`depth.backend: sgbm`. Every documented command passes
`--config config/argus.yaml`, so the drift stayed hidden. Anything run without
the flag used the old rotations and a CPU depth backend.

**Impact.** A calibration captured without `--config` would have baked in the
wrong orientation. A runtime started the same way would have chosen a backend
the production rules forbid.

**Options considered.** Delete the deployed copy; always require `--config`;
sync the deployed copy from the repo template and make reports cite the file
they loaded.

**Resolution.** Backed up the old file to
`/opt/argus/config/argus.yaml.bak-2026-08-13` and replaced it with the repo
template. `ArgusConfig.source` now records which YAML was loaded, and doctor
reports include it.

**Lesson / consequence.** Two config files with one silent default is one too
many. Reports must say which configuration produced them.

## 2026-10-02 M3 — the 256 px image cap made Gemma refuse to look

**Hurdle / problem.** With `image_max_side: 256`, Gemma answered "I cannot see
what is in front of you as I am a text-based AI" to a describe request on a
clear room frame. A 256 px image is only ~15 vision tokens (one token = 48x48
px), and llama.cpp upscales it to its 40-token minimum.

**Impact.** The headline slow-loop question returned a refusal. The earlier
fast describe latency came from starving the model of pixels.

**Options considered.** Raise the pixel cap; set a token budget per task;
change the prompt.

**Resolution.** The client now scales each frame to a token budget: 280 for
describe/find, 560 reserved for reading. The server runs
`--image-max-tokens 560 -ub 1024 -b 1024`. Describe then answers correctly.
The measured cost is 2.2 s warm instead of 1.3 s.

**Lesson / consequence.** A latency number means nothing without the answer
it produced. Benchmarks now store every answer next to its timing.

## 2026-10-02 M3 — Gemma memory: QAT weights and an audio-free projector

**Hurdle / problem.** The resident stack sat near 6.7 GB of 7.6 GB. The Gemma
projector was 985.7 MB, of which 612 MB was an audio encoder ARGUS never uses
(Whisper does speech). It also ran on the CPU (`--no-mmproj-offload`).

**Impact.** Little headroom for depth, detector and OCR, plus swap under load.

**Options considered.** Gemma E4B (rejected by research: ~6 GB);
Q4_K_M with the full projector; QAT UD-Q4_K_XL; a stripped projector, on CPU
or GPU.

**Resolution.** Each change was measured separately
(`reports/vlm-m3-steps-2026-10-02.json`, `reports/vlm-m3-qat-2026-10-02.json`,
prompt cache disabled so every run encodes the image).
- Vision-only projector: 373.4 MB.
- GPU projector: halves image encode time.
- QAT UD-Q4_K_XL (SHA matches the Hugging Face etag): 0.5 GB smaller and faster
  on all three tasks with the same answers.

Production now uses all three. `--cache-ram 0`, one slot and flash attention
are unchanged.

**Lesson / consequence.** Inspect model files before budgeting memory for them.
Half of the vision projector was a different modality.

## 2026-10-02 M3 — grounding no longer needs torch at runtime

**Hurdle / problem.** Every label embedding went through CPU CLIP. That loaded
torch: 1.67 GB RSS and 11.6 s of warm-up.

**Resolution.** A 401-label household and Dhaka-street vocabulary is embedded
offline with the same CLIP call; the build checks parity to 0.0. The runtime
looks labels up (with plural fallback) and loads CLIP only for unknown words.
Grounding warm-up: 353 MB RSS, 1.6 s, torch never imported. The warm label
"object" was initially missing and silently pulled CLIP back in; a test now
asserts torch stays unloaded.

## 2026-10-02 M3 — benchmarks run under over-current throttling

**Hurdle / problem.** The user saw "system throttled due to over current"
pop-ups. `soctherm` channel oc3 recorded 700–965 events during each Gemma
benchmark at MAXN_SUPER with GR3D at 99%. Idle draw is ~6 W and the counter
does not move at idle.

**Impact.** This protects the hardware and does not damage it. But clocks drop,
so the GPU latencies above are pessimistic and noisy, and a busy VLM can slow
the safety loop that shares the GPU.

**Resolution.** Doctor and benchmark reports now record the OC counter delta.
A 25 W vs MAXN_SUPER comparison is planned for M4 depth benchmarking and needs
the user's sudo. The user was asked to confirm the board runs on the original
19 V barrel supply.

## 2026-10-02 M0 — three cameras up; ports, left/right and frame rate chosen by measurement

**Hurdle / problem.** The user reported "all 3 cameras connected", but the
kernel had enumerated only one; the other two showed no USB events at all, so
they had no electrical contact. Once reseated, they landed on different hub
ports than the config expected (B0495s on 3 and 4, B0459 on 2). Snapshot
parallax then showed left/right inverted: the near person shifted ~380 px
between images and the far fan only ~180 px, in the direction that means the
port-4 camera is physically left. The far fan's ~180 px offset also reveals
roughly 15 degrees of relative yaw between the two stereo cameras.

**Impact.** A calibration with swapped roles produces negative disparities.
A 15 degree yaw mismatch costs most of the stereo overlap and is far outside
the ~0.1 degree/px budget.

**Resolution.**
- A spoken hot-plug watcher announced each camera and its link speed in the
  headset as it was plugged in.
- Roles were rebound to the measured ports (left 2-1.4, right 2-1.3,
  wide 2-1.2).
- `argus doctor` is all green.
- Bandwidth matrix (`reports/usb-bandwidth-2026-10-02-2112.json`): the stereo
  pair holds 79 fps even while the wide camera streams 1080p30. The wide
  camera reopens in 0.47 s, so it stays streaming rather than adding half a
  second to every query.
- Capture CPU: 73 / 111 / 152 % of one core at 30 / 60 / 80 fps.
- Stereo set to 60 fps: worst-case pair skew 8.3 ms, so ~2.5 px disparity
  error at 30°/s head rotation instead of ~5 px.
- The mount must be made parallel (M1/M2) before calibration.

**Lesson / consequence.** "Connected" is a claim about cables. Enumeration is
the evidence, and the kernel log distinguishes "no contact" from "USB 2
fallback".

## 2026-10-02 M0 — skew test OOM-killed while the user was wearing the rig

**Hurdle / problem.** The first live `doctor --skew-test` was killed (exit
137). The kernel's global OOM killer also took a VS Code process. The test
kept every raw frame for decoding afterwards: up to 4000 per camera at
~1.7 MB each, beside the 3 GB resident llama-server.

**Impact.** The person wearing the rig was left mid-test. The crash was
announced in the headset immediately, but the guided tool had failed at the
one moment it was supposed to guide.

**Resolution.** A `TimecodeSampler` per camera decodes the newest frame on its
own thread and keeps only (arrival time, decoded ms). Verified on the live
pair: RSS flat at 96 MB over 20 s, ~60 decode attempts/s per camera at 60 fps.

**Lesson / consequence.** On an 8 GB shared-memory board, every measurement
tool needs a memory bound. Buffer results, not images.

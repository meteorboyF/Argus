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

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

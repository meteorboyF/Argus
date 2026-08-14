# ARGUS target architecture

**TARGET DESIGN — NOT IMPLEMENTATION PROOF.** `STATUS.md` is authoritative for
what exists and has been verified.

## Two-Speed system

### Fast safety loop

Target rate: 10–15 Hz, always running independently of speech and Gemma.

```text
AR0234 left/right
  -> timestamp/skew validation
  -> calibrated rectification
  -> GPU stereo depth (TensorRT/CUDA)
  -> rule-based corridor/drop/approach/time-to-collision rules
  -> SLAM pose and tracking quality
  -> priority safety speech/haptic output
```

The decision layer is deterministic and auditable. A neural language model must
never be in this path. GPU depth is permitted because it estimates geometry;
hazard decisions remain explicit rules. Invalid, stale, uncalibrated, or
degraded depth is an UNKNOWN state, never CLEAR.

### Slow interaction loop

```text
ARGUS wake word -> CPU STT -> fresh IMX477P frame
  -> mandatory face and sensitive-text privacy gate
  -> Gemma 4 E2B vision agent via CUDA llama.cpp
  -> optional find_object(name) call
  -> YOLO-World TensorRT grounding
  -> calibrated wide-to-stereo projection and robust depth sampling
  -> concise answer -> CPU Piper speech
```

Every image consumer after capture receives only the gated frame. Gate
initialization failure, inference exception, or invalid output cancels the image
query. Grounded distance is included only when cross-camera calibration and a
fresh valid depth region support it.

## Resource ownership

- GPU: Gemma decoder/vision processing, YOLO-World, stereo depth.
- CPU: capture/orchestration, wake word, STT, TTS, rule evaluation.
- Shared memory target: resident stack at or below approximately 5 GB, proven by
  measurement rather than the proposal estimate.
- Gemma: Q4_K_M, context 2048, one slot, all feasible layers offloaded, flash
  attention enabled only after correctness is verified.
- YOLO-World: FP16 initially; INT8 only after clean calibration evidence.

## Safety boundaries

- Production startup requires verified GPU depth and GPU grounding.
- No stereo calibration: no spoken metric/danger conclusion from stereo.
- No verified wide-to-stereo transform: direction only, no object distance.
- Safety speech preempts ordinary output and stale warnings are discarded.
- Loss of calibration, camera, depth, or pose produces an explicit degraded
  state and conservative user message.

## Intended modules

Existing modules may be replaced internally while retaining clear boundaries:

- `cameras`: stable device identity, transforms, timestamps and health.
- `depth`: calibrated GPU disparity/depth and validity metadata.
- `safety`: pure deterministic rules over timestamped geometry.
- `privacy`: fail-closed face/text transformation and audit metadata.
- `grounding`: TensorRT open-vocabulary contract and detections.
- `agent`: gated-image reasoning and bounded tool protocol.
- `speech`: CPU wake/STT/TTS with priority and cancellation.
- `slam`: future pose/tracking-quality backend.
- `orchestrator`: lifecycle and explicit data contracts between loops.


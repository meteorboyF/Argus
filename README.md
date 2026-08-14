# ARGUS

ARGUS is a research prototype for on-device assistance for blind and visually
impaired users, targeting an NVIDIA Jetson Orin Nano Super 8 GB. The intended
product has an independent, rule-based hazard loop and a speech-triggered visual
assistant. It is not yet a validated mobility aid.

Read [STATUS.md](STATUS.md) first. It is the authority on what works today.
[ARCHITECTURE.md](ARCHITECTURE.md) describes the target, not completed work.

## Current honest state

- The three-camera capture, normalization, USB-role discovery, stereo
  calibration tool, synthetic safety rules, privacy-gated Gemma client, and
  speech components exist as prototype code.
- Gemma 4 E2B has run multimodal inference through a native CUDA llama.cpp build
  on this Jetson after upgrading to L4T R36.5.2.
- YOLO-World grounding runs through a device-built FP16 TensorRT engine with a
  runtime single-label embedding; physical positive-object validation is pending
  and its vocabulary contract is unverified.
- GPU stereo depth is not deployed. No stereo calibration file or RAFT engine is
  present. Production startup therefore intentionally fails closed.
- Face blur exists; sensitive-text handling does not.
- SLAM, incoming-vehicle time-to-collision logic, and calibrated wide-to-stereo
  projection do not exist.
- Object distance is deliberately omitted until cross-camera calibration is
  implemented. The old proportional pixel mapping was unsafe.
- The full test suite is not green: camera, safety, and calibration-health tests
  pass, while the speech-priority suite hangs during shutdown.

## Safe commands that work now

From the repository root:

```bash
python3 -m argus selftest
python3 -m argus --config config/argus.yaml baseline --output /tmp/argus-baseline.json
python3 -m argus preview
python3 scripts/calibrate_stereo.py --help
pytest -q tests/test_cameras.py tests/test_safety.py tests/test_calib_health.py
```

`python3 -m argus run` is a production command and is expected to refuse startup
until the required GPU depth and grounding paths are implemented and verified.
Do not disable the fail-closed settings for a wearable demonstration.

## Hardware and deployed artifacts

- 2 x Arducam B0495/AR0234 USB global-shutter cameras: stereo safety/depth.
- 1 x Arducam B0459/IMX477P USB wide camera: scene questions and grounding.
- USB microphone and headset.
- Jetson Orin Nano Super 8 GB, L4T R36.5.2, MAXN_SUPER.
- Device assets live under `/opt/argus`; exact versions and hashes are recorded
  in [STATUS.md](STATUS.md).

## Active repository map

```text
argus/                  runtime package
config/argus.yaml       production configuration template
scripts/                active setup, calibration, model and engine helpers
tests/                  off-device unit tests
historical/             archived evidence; never use as current instructions
README.md               orientation and honest runnable surface
ARCHITECTURE.md         target Two-Speed architecture
STATUS.md               current implementation truth
AGENT_HANDOFF.md        exact continuation instructions
DECISION_LOG.md         HCI engineering hurdles and decisions
```

The PDF and DOCX under `historical/design-intent/` begin with an explicit
DESIGN-INTENT notice. They are retained as proposal evidence, not proof.

## Non-negotiable rules

1. Run as user `argus`; use `sudo` only for the specific system operation.
2. Build TensorRT engines and llama.cpp on this Jetson.
3. Use the JetPack-matched NVIDIA PyTorch build, never generic PyPI Torch.
4. No image reaches Gemma after a privacy-gate failure or exception.
5. No calibration means no spoken metric/danger inference from stereo depth.
6. Production requires GPU depth and GPU grounding; it never silently falls
   back to CPU SGBM or Ultralytics PyTorch.
7. No verified wide-to-stereo projection means no object distance is spoken.
8. Optional `/etc/fstab` mounts use `nofail` and are validated before reboot.
9. Complete one feature, verify it, update all three living status records,
   commit it, and stop for confirmation.

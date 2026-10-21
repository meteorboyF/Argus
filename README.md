# ARGUS

ARGUS is a research prototype for on-device assistance for blind and visually
impaired users, targeting an NVIDIA Jetson Orin Nano Super 8 GB. The intended
product has an independent, rule-based hazard loop and a speech-triggered visual
assistant. It is not yet a validated mobility aid.

Read [STATUS.md](STATUS.md) first. It is the authority on what works today.
[ARCHITECTURE.md](ARCHITECTURE.md) describes the target, not completed work.

## Current honest state (2026-10-21)

- **Talks.** Say "Argus, what's in front of me?" (or "Argus" → "Yes?" →
  request). A rule router plus Gemma 4 E2B routing drives the skills:
  - describe (privacy-gated image),
  - find an object,
  - read text / answer about it,
  - seat guidance,
  - small talk,
  - shut down.

  Voice turns take 1.6 s (chat) to 3 s (describe).
- **Reads**, with a coach that speaks like a sighted helper (light, glare,
  blur, position). It works on clean signs. A real book cover failed on
  optics: the wide camera's lens needs a one-time refocus
  (`doctor --focus`).
- **Safety loop is built but silent.**
  - Built: corridor hazards (head-level, drop-off), approaching person/vehicle
    TTC, a speech arbiter where hazards cut everything.
  - Silent: metric speech stays off until the stereo pair is calibrated (no
    calibration file yet).
  - Open safety issue: the fast loop stalls up to 0.7 s under the full demo
    stack.
- **Demo/study ready:** first-person view with captions and hazard banner,
  MP4 recording with ARGUS's audio, JSONL session logs, study metrics script.
- Not a validated mobility aid.

## Safe commands that work now

From the repository root:

```bash
python3 -m pytest -q tests/                        # 295 tests
python3 -m argus doctor                            # rig check with spoken summary
scripts/run_llama_server.sh &                      # Gemma on :8080
DISPLAY=:0 python3 -m argus run --demo --record reports/sessions/demo.mp4
DISPLAY=:0 python3 -m argus run --dashboard --no-mic --ask-file /tmp/ask.txt
python3 -m argus align --scene-m 3                 # spoken stereo mount alignment
python3 scripts/calibrate_stereo.py --coach --square-mm N --baseline-mm B
python3 -m argus doctor --focus                    # refocus the wide camera for reading
python3 scripts/summarise_session.py reports/sessions/<log>.jsonl
```

Close the demo with Esc twice or "Argus, shut down". Production runs stay
fail-closed: never disable those settings for a wearable demonstration.

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
docs/FINDINGS_2026-10.md  device findings with evidence (read before changing hardware paths)
docs/prompts/           prompts that drive agent sessions
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

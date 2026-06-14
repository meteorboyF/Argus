# ARGUS — Hardware & Wearable Assembly

The physical build: three cameras mounted on a 3D-printed glasses frame, wired to
the Jetson, with a mic and bone-conduction audio out.

---

## Bill of materials

| Part | Spec | Role |
|---|---|---|
| Compute | NVIDIA Jetson Orin Nano Super (8 GB) | All inference |
| Stereo cameras | 2× Arducam AR0234 (2.3 MP, global shutter, USB 3.0) | Depth |
| Wide camera | 1× Arducam IMX477P (12 MP, USB 3.0, M12 wide lens) | Scene / grounding |
| Microphone | USB mic (16 kHz mono is enough) | Wake word + commands |
| Audio out | Bone-conduction headset or small speaker | TTS guidance |
| Power | USB-C PD power bank (≥ 30 W, ≥ 20 000 mAh) | Mobile power |
| Frame | 3D-printed glasses frame + camera mounts | Wearable |
| Cabling | Short USB 3.0 cables; optional powered USB hub | Connectivity |

---

## Camera layout on the frame

```
        [ AR0234 L ]························[ AR0234 R ]
              \\           baseline B          //
               \\                             //
                \\        [ IMX477P ]        //
                 \\        (wide, center)   //
                  \\                        //
                    ====  glasses front  ====
```

- **Stereo pair (AR0234 L/R):** mount at the **same height**, lenses **parallel**,
  separated by a fixed **baseline** B (≈ 6–10 cm). Rigidity matters more than the
  exact distance — any flex changes depth scale. Global shutter means no motion
  skew while walking.
- **Wide camera (IMX477P):** mount **centered**, between/below the pair, facing
  straight ahead. This is the frame the agent and YOLO-World see.
- Keep all three **coplanar and forward-facing**. Note the chosen baseline; you'll
  confirm it during calibration.

---

## Design notes for the 3D print

- **Stereo rigidity is the #1 requirement.** Print the L/R mounts as part of a
  single stiff cross-member (or add a metal spine) so B can't change. Re-calibrate
  if the frame is ever flexed or re-seated.
- **Strain-relief the USB cables** at the temples so tugging doesn't shift a camera.
- **Vibration isolation** is unnecessary with global-shutter AR0234, but avoid
  mounting cameras on thin cantilevered arms that resonate.
- Leave a **service gap** to reach lens focus rings (the M12 IMX477P focus is manual).
- Route all three USB cables to a **short pigtail** that reaches the Jetson/pack
  worn on a belt or in a pocket.

---

## Wiring & USB bandwidth

Three USB 3.0 cameras can saturate a single controller. On the Jetson:
- Spread cameras across **different USB buses** where possible.
- The runtime requests **MJPG** to reduce bandwidth (see `argus/cameras.py`).
- If a camera intermittently drops, use a **powered** USB 3.0 hub.

Confirm enumeration on-device:
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --all   # inspect a specific camera
```

---

## After assembly

1. Note the physical **baseline** (cm) between the AR0234 lenses.
2. Run `scripts/calibrate_stereo.py` and record `baseline_m` / `focal_px`.
3. Put those into `/opt/argus/config/argus.yaml` under `depth:`.
4. Sanity-check depth: `python -m argus run --no-audio` and walk toward a wall —
   the fast loop should warn at the configured distances.

See [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md) for the software bring-up.

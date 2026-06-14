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

The ARGUS prototype mounts the cameras on a goggle/ski-mask frame:

```
   [ AR0234 L ]                              [ AR0234 R ]
       \  toed outward on the curved sides  /
        \         [ IMX477P wide ]         /
         \         (top center)           /
          ====  goggle front (curved)  ====
```

- **Stereo pair (AR0234 L/R):** on the **curved left/right sides** of the goggle,
  **toed outward**, with a **wide baseline**. They are deliberately **not**
  parallel or coplanar — that's fine. **Full stereo calibration + rectification**
  (see below) measures the actual geometry and corrects for it in software.
- **Wide camera (IMX477P):** mounted **top-center**, facing straight ahead. This
  is the frame the agent and YOLO-World see.
- **The one thing that must not change after calibration: rigidity.** Once you
  calibrate, the L/R cameras must stay fixed relative to each other. Any flex of
  the frame invalidates the calibration — re-run it if the rig is bent or a
  camera is re-seated. Global-shutter AR0234 means no motion skew while walking.

> **Why calibration matters here:** plain stereo math assumes parallel,
> row-aligned cameras. Your toed-out mounting violates that, so raw disparity
> would be wrong. `scripts/calibrate_stereo.py` measures each camera's intrinsics
> and the real rotation/translation between them, then computes rectification maps
> + the Q reprojection matrix. `argus/depth.py` loads these and rectifies every
> frame before matching — so depth is correct for *your* exact mounting, whatever
> the angle.

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

1. Print a flat checkerboard (glue it to stiff card so it stays flat).
2. Run the smart calibrator — it auto-detects the camera pair and board, then
   auto-captures as you move the board around the field of view:
   ```bash
   python scripts/calibrate_stereo.py --square-mm 25
   ```
   It saves rectification maps + Q + baseline/focal to
   `/opt/argus/config/stereo_calib.npz` and reports quality (RMS + vertical
   alignment error). Re-run until RMS < ~0.6 px and vertical error < ~1 px.
3. No config edits needed — `argus/depth.py` auto-loads the `.npz` and rectifies
   every frame. (`baseline_m`/`focal_px` are written in too, for reference.)
4. Sanity-check depth: `python -m argus run --no-audio` and walk toward a wall —
   the fast loop should warn at the configured distances.

> If the goggle frame ever flexes or a camera shifts, **re-run calibration** —
> the rectification is only valid for the geometry it was measured on.

See [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md) for the software bring-up.

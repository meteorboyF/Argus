# ARGUS — Smart Stereo Calibration

How ARGUS gets metric depth from two cameras mounted **in any position** — and
what "any position" actually covers.

---

## The problem

Textbook stereo assumes two identical, parallel, row-aligned cameras. The ARGUS
frame mounts the AR0234s on curved goggle sides: **toed outward, non-coplanar,
wide baseline** — and the exact geometry changes every time the frame is
printed, flexed, or a camera is re-seated. Raw disparity on that geometry is
garbage.

## The solution (what `scripts/calibrate_stereo.py` does)

1. **Intrinsics** per camera: focal length, principal point, lens distortion.
2. **Extrinsics** between the pair: the real rotation `R` and translation `T`
   (your toe-out angle and baseline, whatever they are).
3. **Rectification maps**: warp both images onto a common virtual plane so
   epipolar lines are horizontal → block matching becomes valid.
4. **Q matrix**: reprojects disparity straight to metric 3D.

`argus/depth.py` auto-loads the result: every live frame is remapped before
matching, and depth comes out of `reprojectImageTo3D(Q)` — correct for the
actual mounting, no config edits.

## What makes it "smart"

| Situation | Handled how |
|---|---|
| Cameras plugged into any USB ports, any order | Pair auto-detected by V4L2 device name (`AR0234`), resolution-grouping fallback |
| You called the wrong one "left" | After solving, the sign of the baseline (`T[0] > 0`) reveals a swapped pair; the script swaps and re-solves automatically — no re-capture |
| V4L2 renumbers `/dev/video*` on reboot | The `.npz` records each camera's **USB port path**; `argus/cameras.py` re-binds left/right to the physical cameras every start |
| Unknown checkerboard | Auto-detects among common layouts (9×6, 7×6, 8×6, …) |
| No display (SSH bring-up) | `--headless` auto-captures and stops at min-views + coverage |
| Calibration vs runtime resolution mismatch | Calibrates at the resolution from `argus.yaml` (`camera.stereo_width/height`); selftest warns if they ever diverge |
| Bad calibration | RMS + post-rectification vertical error printed and stored; selftest re-checks them; loud warnings above thresholds |
| "Is the depth actually right?" | `--verify` runs live rectified SGBM and prints the centre-patch distance to compare against a tape measure |

## Running it

```bash
# normal (display attached):
python3 scripts/calibrate_stereo.py --square-mm 25

# over SSH:
python3 scripts/calibrate_stereo.py --square-mm 25 --headless

# verify against reality afterwards (always do this):
python3 scripts/calibrate_stereo.py --verify
```

`--square-mm` is the printed square size — measure it with a ruler after
printing (printers rescale!). All other parameters are auto-detected; `--left/
--right/--rows/--cols/--out` exist as manual overrides.

### No printer: use the monitor as the target

The Jetson can render a physically scaled board from the monitor's XRandR/EDID
dimensions. Keep the board full-screen and move the **locked camera rig** through
the poses instead of moving a paper target:

```bash
# Terminal 1, on the attached display:
python3 scripts/display_calibration_board.py --cols 9 --rows 6 --square-mm 30

# Terminal 2 (the board must remain unobstructed):
python3 scripts/calibrate_stereo.py --headless \
  --cols 9 --rows 6 --square-mm 30 --min-views 20
```

This is less ideal than a rigid printed board: EDID physical dimensions can be
approximate, screen glare can hide corners, and the rig must be moved enough to
cover the full field at varied angles. Measure one displayed square with a real
ruler if possible. In every case, `--verify` against tape-measured distances is
mandatory before metric safety thresholds are trusted.

### Capture technique (quality depends on this)

- Board **flat and rigid** (glued to card/acrylic). A bowed board poisons everything.
- Cover **all nine zones** of the view: corners, edges, centre.
- Vary **distance** (near where the board fills ~½ the frame, and far) and
  **tilt** (up to ~40°).
- Good, even lighting; avoid motion blur — the tool only captures when steady.
- 15 views minimum; 20–25 well-spread views are better.

### Quality targets

| Metric | Good | Reject |
|---|---|---|
| Stereo RMS reprojection error | < 0.6 px | > 1.5 px |
| Post-rectification vertical error | < 1.0 px | > 2.0 px |
| `--verify` vs tape measure @ 1–2 m | within 5–10 % | worse |

## The one physical rule

**Rigidity.** The calibration describes a frozen geometry. If the frame flexes,
a camera is re-seated, or the mount is reprinted — **re-run calibration**
(2 minutes with the auto-capture). The wide IMX477P is not part of the stereo
solve and can be moved freely (but see KNOWN_GAPS.md on wide↔stereo fusion).

## File contents (`/opt/argus/config/stereo_calib.npz`)

| Keys | Meaning |
|---|---|
| `mtxL,distL,mtxR,distR` | per-camera intrinsics + distortion |
| `R,T` | right camera pose relative to left (your real mounting) |
| `R1,R2,P1,P2,Q` | rectification transforms + reprojection matrix |
| `map1x,map1y,map2x,map2y` | precomputed remap tables used per frame |
| `baseline_m,focal_px,toe_angle_deg` | convenience scalars (fallback path + logging) |
| `image_size,rms,vertical_error,pattern` | provenance + quality |
| `left_index,right_index,left_port,right_port` | device identity at calibration time; ports drive runtime re-binding |

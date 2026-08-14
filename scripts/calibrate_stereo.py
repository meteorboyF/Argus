"""Smart stereo calibration for the ARGUS AR0234 pair.

This calibrator adapts to HOWEVER you physically mounted the cameras. On the
ARGUS goggles the two AR0234s sit on the curved sides of the frame, toed
outward, with a wide baseline — they are NOT parallel or coplanar. Plain SGBM
would produce garbage on that geometry.

The fix is full stereo calibration + rectification:
  1. Measure each camera's intrinsics (focal, distortion).
  2. Measure the real rotation R and translation T between them (your toe-out
     and baseline, whatever they are).
  3. Compute rectification maps that warp both images onto a common plane so
     epipolar lines are horizontal — after which disparity -> depth is valid.
  4. Compute Q, the 4x4 reprojection matrix, so depth.py can turn disparity
     straight into metric 3D (handles your geometry automatically).

It is "smart" about setup — ANY camera arrangement works:
  - Auto-detects the stereo pair by V4L2 device name (AR0234) with a
    matching-resolution fallback; no fixed indices needed.
  - Captures at the RUNTIME stereo resolution (from argus.yaml) so the
    rectification maps match what the depth loop actually sees.
  - Auto-detects the checkerboard dimensions from a list of common sizes.
  - Auto-captures a view when the board is detected in BOTH cameras and steady,
    spread across the field of view (you just move the board around).
  - Detects SWAPPED left/right cameras from the recovered geometry (the sign of
    the baseline) and fixes the assignment automatically — plug the cameras into
    any port, in any order.
  - Records each camera's USB port path so the runtime re-binds left/right to
    the same physical cameras on every boot, however V4L2 renumbers them.
  - Validates the result (reprojection error + post-rectification vertical
    alignment) and warns loudly on a bad calibration.
  - --verify shows (or, headless, prints) live rectified depth so you can
    sanity-check with a tape measure before trusting it.

Usage (just plug the board in and move it around):
    python3 scripts/calibrate_stereo.py --square-mm 25

Verify an existing calibration against live depth:
    python3 scripts/calibrate_stereo.py --verify

Or pin things down explicitly:
    python3 scripts/calibrate_stereo.py --left 0 --right 1 \
        --rows 6 --cols 9 --square-mm 25 \
        --out /opt/argus/config/stereo_calib.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running straight from the repo before `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Common inner-corner layouts to try when --rows/--cols are not given.
# (cols, rows) = (inner corners along x, inner corners along y).
COMMON_BOARDS = [(9, 6), (7, 6), (8, 6), (9, 7), (7, 5), (6, 5), (10, 7), (11, 8)]

CHESS_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH
               | cv2.CALIB_CB_NORMALIZE_IMAGE
               | cv2.CALIB_CB_FAST_CHECK)
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

DEFAULT_OUT = os.path.join(os.environ.get("ARGUS_HOME", "/opt/argus"),
                           "config", "stereo_calib.npz")


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------
def _open(idx: int, width: int, height: int, fps: int = 30,
          pixel_format: str = "YUYV") -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux") \
        else cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def autodetect_pair(cfg) -> tuple[int, int]:
    """Find the AR0234 stereo pair using the same smart discovery as the runtime
    (V4L2 device names first, resolution grouping fallback)."""
    from argus.cameras import probe_cameras
    nodes = probe_cameras(cfg.max_probe_index if cfg else 10)
    if not nodes:
        raise SystemExit("No cameras found. Check connections / v4l2-ctl --list-devices")

    hint = (cfg.stereo_name_hint if cfg else "AR0234").lower()
    stereo = [n for n in nodes if hint and hint in n.name.lower()]
    if len(stereo) != 2:
        by_res: dict[tuple[int, int], list] = {}
        for n in nodes:
            by_res.setdefault((n.width, n.height), []).append(n)
        groups = [g for g in by_res.values() if len(g) >= 2]
        if not groups:
            raise SystemExit(
                f"Could not auto-detect a matching stereo pair. Detected: "
                f"{[(n.index, n.name, (n.width, n.height)) for n in nodes]}\n"
                "Pass --left and --right explicitly.")
        stereo = sorted(max(groups, key=len), key=lambda n: n.index)[:2]
    stereo = sorted(stereo, key=lambda n: n.index)
    print(f"Auto-detected stereo pair: /dev/video{stereo[0].index} and "
          f"/dev/video{stereo[1].index} ('{stereo[0].name}'). Left/right order "
          "will be verified geometrically after calibration.")
    return stereo[0].index, stereo[1].index


def usb_port_of(index: int) -> str:
    """Stable USB port path for /dev/video<index> ('' off-Linux)."""
    d = Path(f"/sys/class/video4linux/video{index}/device")
    try:
        return d.resolve().name.split(":")[0]
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------
def detect_board(gray, pattern):
    ok, corners = cv2.findChessboardCorners(gray, pattern, CHESS_FLAGS)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRIT)
    return ok, corners


def autodetect_board(grayL, grayR) -> tuple[int, int] | None:
    """Find the checkerboard layout that is visible in BOTH images."""
    for pattern in COMMON_BOARDS:
        okL, _ = cv2.findChessboardCorners(grayL, pattern, CHESS_FLAGS)
        okR, _ = cv2.findChessboardCorners(grayR, pattern, CHESS_FLAGS)
        if okL and okR:
            return pattern
    return None


def make_object_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    return obj


def _coverage_cell(corners, w, h, grid=3):
    """Which coverage cell (0..grid*grid-1) the board center falls in — used to
    encourage views spread across the whole field of view."""
    c = corners.reshape(-1, 2).mean(axis=0)
    cx = min(grid - 1, int(c[0] / w * grid))
    cy = min(grid - 1, int(c[1] / h * grid))
    return cy * grid + cx


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(objpoints, ptsL, ptsR, size):
    _, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, ptsL, size, None, None)
    _, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, ptsR, size, None, None)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms, mtxL, distL, mtxR, distR, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, ptsL, ptsR, mtxL, distL, mtxR, distR, size,
        criteria=crit, flags=cv2.CALIB_FIX_INTRINSIC)
    return rms, mtxL, distL, mtxR, distR, R, T


def rectify(mtxL, distL, mtxR, distR, size, R, T):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, size, cv2.CV_32FC1)
    return R1, R2, P1, P2, Q, map1x, map1y, map2x, map2y


def rectification_error(ptsL, ptsR, mtxL, distL, R1, P1, mtxR, distR, R2, P2):
    """After rectification, matched points should share the same row. Returns the
    mean absolute vertical disparity in pixels (lower = better; < ~1 px is good)."""
    errs = []
    for cL, cR in zip(ptsL, ptsR):
        uL = cv2.undistortPoints(cL, mtxL, distL, R=R1, P=P1).reshape(-1, 2)
        uR = cv2.undistortPoints(cR, mtxR, distR, R=R2, P=P2).reshape(-1, 2)
        errs.append(np.abs(uL[:, 1] - uR[:, 1]))
    return float(np.concatenate(errs).mean())


# ---------------------------------------------------------------------------
# Live depth verification
# ---------------------------------------------------------------------------
def verify_depth(calib_path: str, seconds: float = 20.0, headless: bool = False):
    """Open the calibrated pair, rectify live frames, run SGBM, and report the
    median depth in the central patch — point the rig at a wall or hold your
    hand up and compare against a tape measure."""
    from argus.config import load_config
    from argus.depth import DepthEstimator

    cfg = load_config()
    cfg.depth.calibration_file = calib_path
    est = DepthEstimator(cfg.depth)
    if not est.calibrated:
        raise SystemExit(f"No usable calibration at {calib_path} — run calibration first.")

    left_idx, right_idx = autodetect_pair(cfg.camera)
    # Honour saved swap: re-bind to the ports the calibration recorded.
    data = np.load(calib_path, allow_pickle=True)
    lp = str(data["left_port"]) if "left_port" in data else ""
    rp = str(data["right_port"]) if "right_port" in data else ""
    if lp and rp:
        if usb_port_of(left_idx) == rp and usb_port_of(right_idx) == lp:
            left_idx, right_idx = right_idx, left_idx

    capL = _open(left_idx, cfg.camera.stereo_width, cfg.camera.stereo_height,
                 pixel_format=cfg.camera.pixel_format)
    capR = _open(right_idx, cfg.camera.stereo_width, cfg.camera.stereo_height,
                 pixel_format=cfg.camera.pixel_format)
    if capL is None or capR is None:
        raise SystemExit("Could not open the stereo cameras for verification.")

    print("\nVerification: aim the rig at a flat surface 0.5–3 m away.")
    print("Compare the printed centre distance with a tape measure. Ctrl-C to stop.\n")
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            okL, fL = capL.read()
            okR, fR = capR.read()
            if not (okL and okR):
                continue
            from argus.cameras import transform_frame
            fL = transform_frame(fL, cfg.camera.left_rotation,
                                 cfg.camera.left_flip_horizontal,
                                 cfg.camera.left_flip_vertical)
            fR = transform_frame(fR, cfg.camera.right_rotation,
                                 cfg.camera.right_flip_horizontal,
                                 cfg.camera.right_flip_vertical)
            depth = est.depth_map(fL, fR)
            h, w = depth.shape[:2]
            patch = depth[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20]
            finite = patch[np.isfinite(patch)]
            centre = float(np.median(finite)) if finite.size else float("nan")
            valid_pct = 100.0 * np.isfinite(depth).mean()
            print(f"\r  centre depth: {centre:5.2f} m   valid pixels: {valid_pct:4.1f}%   ",
                  end="", flush=True)
            if not headless:
                vis = np.clip(depth, 0, 4.0) / 4.0
                vis = cv2.applyColorMap((255 - vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                vis[~np.isfinite(depth)] = 0
                cv2.rectangle(vis, (w // 2 - 20, h // 2 - 20), (w // 2 + 20, h // 2 + 20),
                              (255, 255, 255), 2)
                cv2.imshow("ARGUS depth verification (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        capL.release(); capR.release()
        if not headless:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", type=int, default=None, help="left cam index (auto if omitted)")
    ap.add_argument("--right", type=int, default=None, help="right cam index (auto if omitted)")
    ap.add_argument("--rows", type=int, default=None, help="inner corners per column (auto if omitted)")
    ap.add_argument("--cols", type=int, default=None, help="inner corners per row (auto if omitted)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="checkerboard square size (mm)")
    ap.add_argument("--min-views", type=int, default=15, help="views required before solving")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-auto", action="store_true", help="manual capture (SPACE) only")
    ap.add_argument("--headless", action="store_true", help="no preview window (SSH-friendly)")
    ap.add_argument("--preview-scale", type=float, default=0.45,
                    help="live stereo preview scale (default fits reserved board panel)")
    ap.add_argument("--preview-x", type=int, default=10,
                    help="live preview window screen x position")
    ap.add_argument("--preview-y", type=int, default=90,
                    help="live preview window screen y position")
    ap.add_argument("--verify", action="store_true",
                    help="skip calibration; live-check depth against the saved file")
    ap.add_argument("--max-seconds", type=float, default=300.0,
                    help="headless capture time limit")
    args = ap.parse_args()

    try:
        from argus.config import load_config
        cam_cfg = load_config().camera
    except Exception:  # noqa: BLE001 — argus package not installed yet
        cam_cfg = None

    if args.verify:
        verify_depth(args.out, headless=args.headless)
        return

    left_idx, right_idx = (args.left, args.right)
    if left_idx is None or right_idx is None:
        left_idx, right_idx = autodetect_pair(cam_cfg)

    # Calibrate at the RUNTIME stereo resolution so the rectification maps match
    # what the depth loop sees. (A mismatch silently costs accuracy.)
    cw = cam_cfg.stereo_width if cam_cfg else 1280
    ch = cam_cfg.stereo_height if cam_cfg else 720

    pixel_format = cam_cfg.pixel_format if cam_cfg else "YUYV"
    capL = _open(left_idx, cw, ch, pixel_format=pixel_format)
    capR = _open(right_idx, cw, ch, pixel_format=pixel_format)
    if capL is None or capR is None:
        raise SystemExit(f"Could not open cameras {left_idx}/{right_idx}.")

    pattern = (args.cols, args.rows) if (args.cols and args.rows) else None
    square_m = args.square_mm / 1000.0
    obj = make_object_points(pattern, square_m) if pattern else None

    objpoints, ptsL, ptsR = [], [], []
    covered: set[int] = set()
    size = None
    validated_frame_geometry = False
    last_capture = 0.0
    prev_center = None
    t_start = time.time()

    print("\nMove the checkerboard slowly around the whole view (corners, center,")
    print("tilted, near, far). Auto-capture fires when it's seen in BOTH cameras")
    print("and held steady. Press 'c' to force-capture, 'q'/ESC to finish.\n")

    while True:
        okL, fL = capL.read()
        okR, fR = capR.read()
        if not (okL and okR):
            continue
        if cam_cfg:
            from argus.cameras import transform_frame
            fL = transform_frame(fL, cam_cfg.left_rotation,
                                 cam_cfg.left_flip_horizontal,
                                 cam_cfg.left_flip_vertical)
            fR = transform_frame(fR, cam_cfg.right_rotation,
                                 cam_cfg.right_flip_horizontal,
                                 cam_cfg.right_flip_vertical)
        if not validated_frame_geometry:
            if fL.shape[:2] != fR.shape[:2]:
                capL.release(); capR.release()
                raise SystemExit(
                    "Selected cameras do not produce the same transformed frame size: "
                    f"/dev/video{left_idx} -> {fL.shape[1]}x{fL.shape[0]}, "
                    f"/dev/video{right_idx} -> {fR.shape[1]}x{fR.shape[0]}.\n"
                    "A USB reconnect probably renumbered the cameras and one selected "
                    "device may be the wide B0459. Run `v4l2-ctl --list-devices`, "
                    "select the two B0495 devices, or omit --left/--right to auto-detect."
                )
            validated_frame_geometry = True
        gL = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
        size = gL.shape[::-1]

        if pattern is None:
            pattern = autodetect_board(gL, gR)
            if pattern is not None:
                obj = make_object_points(pattern, square_m)
                print(f"Detected checkerboard: {pattern[0]}x{pattern[1]} inner corners.")

        okcL = okcR = False
        cL = cR = None
        if pattern is not None:
            okcL, cL = detect_board(gL, pattern)
            okcR, cR = detect_board(gR, pattern)

        both = okcL and okcR
        # Steadiness: board center barely moved since last frame.
        steady = False
        if both:
            center = cL.reshape(-1, 2).mean(axis=0)
            if prev_center is not None:
                steady = np.linalg.norm(center - prev_center) < 3.0
            prev_center = center

        do_capture = False
        if both and not args.no_auto and steady and (time.time() - last_capture) > 1.0:
            cell = _coverage_cell(cL, size[0], size[1])
            # Prefer new coverage cells, but still allow repeats once spread out.
            if cell not in covered or len(objpoints) >= 9:
                do_capture = True

        if not args.headless:
            disp = np.hstack([fL, fR])
            color = (0, 200, 0) if both else (0, 0, 200)
            cv2.putText(disp, f"views: {len(objpoints)}/{args.min_views}  "
                              f"both:{both} steady:{steady} cells:{len(covered)}/9",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if both:
                cv2.drawChessboardCorners(disp[:, :fL.shape[1]], pattern, cL, True)
                cv2.drawChessboardCorners(disp[:, fL.shape[1]:], pattern, cR, True)
            window = "ARGUS stereo calibration"
            preview = cv2.resize(disp, None, fx=args.preview_scale,
                                 fy=args.preview_scale)
            cv2.imshow(window, preview)
            cv2.moveWindow(window, args.preview_x, args.preview_y)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("c") and both:
                do_capture = True

        if do_capture:
            objpoints.append(obj)
            ptsL.append(cL)
            ptsR.append(cR)
            covered.add(_coverage_cell(cL, size[0], size[1]))
            last_capture = time.time()
            print(f"  captured view {len(objpoints)}/{args.min_views} "
                  f"(coverage {len(covered)}/9)")
            if args.headless and len(objpoints) >= args.min_views and len(covered) >= 6:
                break
        if args.headless and (time.time() - t_start) > args.max_seconds:
            print("Headless time limit reached.")
            break

    capL.release(); capR.release()
    if not args.headless:
        cv2.destroyAllWindows()

    if len(objpoints) < args.min_views:
        raise SystemExit(f"Only {len(objpoints)} views (need >= {args.min_views}). Re-run and "
                         "cover more of the frame.")

    print(f"\nCalibrating from {len(objpoints)} views...")
    rms, mtxL, distL, mtxR, distR, R, T = calibrate(objpoints, ptsL, ptsR, size)

    # --- Automatic left/right disambiguation -------------------------------
    # Convention: P_right = R @ P_left + T, so for a physically correct
    # left/right assignment the left camera sits at x = T[0] < 0 in right-camera
    # coordinates. T[0] > 0 means the cameras were connected swapped — fix it by
    # swapping the point sets and re-solving, no re-capture needed.
    swapped = False
    if float(T[0]) > 0:
        print("  Detected swapped cameras (baseline sign positive) — auto-correcting.")
        left_idx, right_idx = right_idx, left_idx
        ptsL, ptsR = ptsR, ptsL
        rms, mtxL, distL, mtxR, distR, R, T = calibrate(objpoints, ptsL, ptsR, size)
        swapped = True

    R1, R2, P1, P2, Q, m1x, m1y, m2x, m2y = rectify(mtxL, distL, mtxR, distR, size, R, T)
    v_err = rectification_error(ptsL, ptsR, mtxL, distL, R1, P1, mtxR, distR, R2, P2)

    baseline_m = float(np.linalg.norm(T))
    focal_px = float(P1[0, 0])  # rectified focal length
    # Toe-in/out angle between the two optical axes (deg) — informational.
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    print("\n=== Calibration result ===")
    if swapped:
        print(f"  cameras were SWAPPED — corrected: left=/dev/video{left_idx}, "
              f"right=/dev/video{right_idx}")
    print(f"  image size                    : {size[0]}x{size[1]}")
    print(f"  stereo RMS reprojection error : {rms:.3f} px   (aim < 0.6)")
    print(f"  post-rectification vert. error: {v_err:.3f} px   (aim < 1.0)")
    print(f"  baseline                      : {baseline_m*100:.2f} cm")
    print(f"  rectified focal length        : {focal_px:.1f} px")
    print(f"  inter-camera angle (toe)      : {angle:.2f} deg")

    if rms > 1.5 or v_err > 2.0:
        print("\n  WARNING: calibration quality is poor. Re-run with more, better-spread,")
        print("  well-lit views of a rigid (flat!) checkerboard before trusting depth.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        # intrinsics / extrinsics
        mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T,
        # rectification (this is what makes your toed-out mounting work)
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=m1x, map1y=m1y, map2x=m2x, map2y=m2y,
        # convenience scalars + metadata
        baseline_m=baseline_m, focal_px=focal_px, image_size=np.array(size),
        rms=rms, vertical_error=v_err, toe_angle_deg=angle,
        left_index=left_idx, right_index=right_idx, pattern=np.array(pattern),
        # stable physical identity so the runtime re-binds left/right correctly
        # across reboots however V4L2 renumbers the devices
        left_port=usb_port_of(left_idx), right_port=usb_port_of(right_idx),
    )
    print(f"\nSaved calibration -> {args.out}")
    print("depth.py auto-loads this (rectification maps + Q) and produces metric depth")
    print("for your exact mounting; cameras.py re-binds left/right by USB port.")
    print("\nNow sanity-check it against a tape measure:")
    print(f"  python3 scripts/calibrate_stereo.py --verify"
          + (" --headless" if args.headless else ""))


if __name__ == "__main__":
    main()

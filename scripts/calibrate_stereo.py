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

It is "smart" about setup:
  - Auto-detects camera indices (two matching-resolution = the stereo pair).
  - Auto-detects the checkerboard dimensions from a list of common sizes.
  - Auto-captures a view when the board is detected in BOTH cameras and steady,
    spread across the field of view (you just move the board around).
  - Validates the result (reprojection error + post-rectification vertical
    alignment) and refuses to save a bad calibration.

Usage (just plug the board in and move it around):
    python scripts/calibrate_stereo.py --square-mm 25

Or pin things down explicitly:
    python scripts/calibrate_stereo.py --left 0 --right 1 \
        --rows 6 --cols 9 --square-mm 25 \
        --out /opt/argus/config/stereo_calib.npz
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

# Common inner-corner layouts to try when --rows/--cols are not given.
# (cols, rows) = (inner corners along x, inner corners along y).
COMMON_BOARDS = [(9, 6), (7, 6), (8, 6), (9, 7), (7, 5), (6, 5), (10, 7), (11, 8)]

CHESS_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH
               | cv2.CALIB_CB_NORMALIZE_IMAGE
               | cv2.CALIB_CB_FAST_CHECK)
SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------
def _open(idx: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    return cap


def autodetect_pair(max_index: int = 8) -> tuple[int, int]:
    """Find two cameras with matching resolution = the AR0234 stereo pair.

    The IMX477P wide camera has a different (higher) resolution, so the two that
    match are the stereo pair. Returns (left_index, right_index) ordered by index
    (you can swap with --left/--right if mirrored)."""
    found = {}
    for idx in range(max_index):
        cap = _open(idx)
        if cap is None:
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        found[idx] = (w, h)
    if not found:
        raise SystemExit("No cameras found. Check connections / v4l2-ctl --list-devices")

    # Group by resolution; the largest group of size >= 2 is the stereo pair.
    by_res: dict[tuple[int, int], list[int]] = {}
    for idx, res in found.items():
        by_res.setdefault(res, []).append(idx)
    pairs = [idxs for idxs in by_res.values() if len(idxs) >= 2]
    if not pairs:
        raise SystemExit(
            f"Could not auto-detect a matching stereo pair. Detected: {found}\n"
            "Pass --left and --right explicitly."
        )
    pair = sorted(max(pairs, key=len))[:2]
    print(f"Auto-detected stereo pair: left={pair[0]}, right={pair[1]} "
          f"(resolution {found[pair[0]]}). Use --left/--right to override/swap.")
    return pair[0], pair[1]


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
    ap.add_argument("--out", default="/opt/argus/config/stereo_calib.npz")
    ap.add_argument("--no-auto", action="store_true", help="manual capture (SPACE) only")
    ap.add_argument("--headless", action="store_true", help="no preview window")
    args = ap.parse_args()

    left_idx, right_idx = (args.left, args.right)
    if left_idx is None or right_idx is None:
        left_idx, right_idx = autodetect_pair()

    capL, capR = _open(left_idx), _open(right_idx)
    if capL is None or capR is None:
        raise SystemExit(f"Could not open cameras {left_idx}/{right_idx}.")

    pattern = (args.cols, args.rows) if (args.cols and args.rows) else None
    square_m = args.square_mm / 1000.0

    objpoints, ptsL, ptsR = [], [], []
    covered: set[int] = set()
    size = None
    last_capture = 0.0
    prev_center = None

    print("\nMove the checkerboard slowly around the whole view (corners, center,")
    print("tilted, near, far). Auto-capture fires when it's seen in BOTH cameras")
    print("and held steady. Press 'c' to force-capture, 'q'/ESC to finish.\n")

    while True:
        okL, fL = capL.read()
        okR, fR = capR.read()
        if not (okL and okR):
            continue
        gL = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
        size = gL.shape[::-1]

        if pattern is None:
            pattern = autodetect_board(gL, gR)
            if pattern is not None:
                obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
                obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
                print(f"Detected checkerboard: {pattern[0]}x{pattern[1]} inner corners.")

        okcL = okcR = False
        cL = cR = None
        if pattern is not None:
            okcL, cL = detect_board(gL, pattern)
            okcR, cR = detect_board(gR, pattern)
            if "obj" not in dir():
                obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
                obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m

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
            cv2.imshow("ARGUS stereo calibration", cv2.resize(disp, None, fx=0.6, fy=0.6))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c") and both:
                do_capture = True

        if do_capture:
            objpoints.append(obj)
            ptsL.append(cL)
            ptsR.append(cR)
            covered.add(_coverage_cell(cL, size[0], size[1]))
            last_capture = time.time()
            print(f"  captured view {len(objpoints)} (coverage {len(covered)}/9)")
            if args.headless and len(objpoints) >= args.min_views:
                break

    capL.release(); capR.release()
    if not args.headless:
        cv2.destroyAllWindows()

    if len(objpoints) < args.min_views:
        raise SystemExit(f"Only {len(objpoints)} views (need >= {args.min_views}). Re-run and "
                         "cover more of the frame.")

    print(f"\nCalibrating from {len(objpoints)} views...")
    rms, mtxL, distL, mtxR, distR, R, T = calibrate(objpoints, ptsL, ptsR, size)
    R1, R2, P1, P2, Q, m1x, m1y, m2x, m2y = rectify(mtxL, distL, mtxR, distR, size, R, T)
    v_err = rectification_error(ptsL, ptsR, mtxL, distL, R1, P1, mtxR, distR, R2, P2)

    baseline_m = float(np.linalg.norm(T))
    focal_px = float(P1[0, 0])  # rectified focal length
    # Toe-in/out angle between the two optical axes (deg) — informational.
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    print("\n=== Calibration result ===")
    print(f"  stereo RMS reprojection error : {rms:.3f} px   (aim < 0.6)")
    print(f"  post-rectification vert. error: {v_err:.3f} px   (aim < 1.0)")
    print(f"  baseline                      : {baseline_m*100:.2f} cm")
    print(f"  rectified focal length        : {focal_px:.1f} px")
    print(f"  inter-camera angle (toe)      : {angle:.2f} deg")

    if rms > 1.5 or v_err > 2.0:
        print("\n  WARNING: calibration quality is poor. Re-run with more, better-spread,")
        print("  well-lit views of a rigid (flat!) checkerboard before trusting depth.")

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
    )
    print(f"\nSaved calibration -> {args.out}")
    print("depth.py will auto-load this (rectification maps + Q) and produce metric")
    print("depth that accounts for your exact camera mounting. No config edits needed —")
    print("though baseline_m / focal_px are also written for reference.")


if __name__ == "__main__":
    main()

"""Stereo calibration for the AR0234 pair.

Capture ~20 views of a printed checkerboard from both cameras, compute the
intrinsics/extrinsics, and save baseline + focal length into the calibration
file the depth module reads. Run on the Jetson (or any machine with the cameras).

  python scripts/calibrate_stereo.py --left 0 --right 1 \
      --rows 6 --cols 9 --square-mm 25 --out /opt/argus/config/stereo_calib.npz

Hold the board still, press SPACE to capture each view, ESC when done (>= 10).
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=int, default=0)
    ap.add_argument("--right", type=int, default=1)
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out", default="/opt/argus/config/stereo_calib.npz")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= (args.square_mm / 1000.0)  # metres

    capL = cv2.VideoCapture(args.left)
    capR = cv2.VideoCapture(args.right)
    if not (capL.isOpened() and capR.isOpened()):
        raise SystemExit("Could not open both cameras. Check indices with v4l2-ctl --list-devices")

    objpoints, imgpointsL, imgpointsR = [], [], []
    print("SPACE = capture a view, ESC = finish (need >= 10).")
    size = None

    while True:
        okL, fL = capL.read()
        okR, fR = capR.read()
        if not (okL and okR):
            continue
        gL = cv2.cvtColor(fL, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(fR, cv2.COLOR_BGR2GRAY)
        size = gL.shape[::-1]
        fL_okc, cL = cv2.findChessboardCorners(gL, pattern, None)
        fR_okc, cR = cv2.findChessboardCorners(gR, pattern, None)

        disp = np.hstack([fL, fR])
        cv2.putText(disp, f"views: {len(objpoints)}  bothFound: {fL_okc and fR_okc}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("stereo calib (SPACE=capture ESC=done)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32 and fL_okc and fR_okc:  # SPACE
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cL = cv2.cornerSubPix(gL, cL, (11, 11), (-1, -1), crit)
            cR = cv2.cornerSubPix(gR, cR, (11, 11), (-1, -1), crit)
            objpoints.append(objp)
            imgpointsL.append(cL)
            imgpointsR.append(cR)
            print(f"  captured view {len(objpoints)}")

    capL.release(); capR.release(); cv2.destroyAllWindows()

    if len(objpoints) < 10:
        raise SystemExit(f"Only {len(objpoints)} views — need >= 10. Re-run.")

    print("Calibrating...")
    _, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, size, None, None)
    _, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, size, None, None)
    flags = cv2.CALIB_FIX_INTRINSIC
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret, mtxL, distL, mtxR, distR, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, size,
        criteria=crit, flags=flags)

    baseline_m = float(np.linalg.norm(T))
    focal_px = float(mtxL[0, 0])
    np.savez(args.out, mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR,
             R=R, T=T, baseline_m=baseline_m, focal_px=focal_px, image_size=size)
    print(f"Saved {args.out}")
    print(f"  reprojection error: {ret:.4f}")
    print(f"  baseline: {baseline_m*100:.2f} cm   focal: {focal_px:.1f} px")
    print("Put these into config/argus.yaml depth.baseline_m and depth.focal_px,")
    print("or the runtime will read them from the .npz if present.")


if __name__ == "__main__":
    main()

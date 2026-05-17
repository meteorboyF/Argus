"""
Stereo calibration for the Arducam AR0234 pair.

Usage (on Jetson / Linux):
    python calibrate_stereo.py --left 0 --right 1 --rows 9 --cols 6 \
                                --square 0.025 --output ../configs/stereo_calib.npz

Capture checkerboard images by pressing SPACE, quit with Q.
At least 15 valid pairs are recommended.
"""
import argparse, sys, time
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="AR0234 stereo calibration")
    p.add_argument("--left",   type=int,   default=0,      help="Left camera index")
    p.add_argument("--right",  type=int,   default=1,      help="Right camera index")
    p.add_argument("--rows",   type=int,   default=9,      help="Checkerboard inner rows")
    p.add_argument("--cols",   type=int,   default=6,      help="Checkerboard inner cols")
    p.add_argument("--square", type=float, default=0.025,  help="Square size in metres")
    p.add_argument("--output", type=str,   default="../configs/stereo_calib.npz")
    p.add_argument("--width",  type=int,   default=640)
    p.add_argument("--height", type=int,   default=480)
    return p.parse_args()


def main():
    args = parse_args()
    board = (args.cols, args.rows)

    obj_pts_template = np.zeros((args.rows * args.cols, 3), np.float32)
    obj_pts_template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    obj_pts_template *= args.square

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    obj_points, pts_l, pts_r = [], [], []

    cap_l = cv2.VideoCapture(args.left,  cv2.CAP_V4L2)
    cap_r = cv2.VideoCapture(args.right, cv2.CAP_V4L2)
    for cap in (cap_l, cap_r):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Press SPACE to capture, Q to quit and calibrate.")
    n = 0
    while True:
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            print("Camera read error.")
            break

        gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        disp = np.hstack([fl, fr])
        cv2.imshow("Stereo (SPACE=capture, Q=done)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            found_l, corners_l = cv2.findChessboardCorners(gl, board)
            found_r, corners_r = cv2.findChessboardCorners(gr, board)
            if found_l and found_r:
                cv2.cornerSubPix(gl, corners_l, (11, 11), (-1, -1), criteria)
                cv2.cornerSubPix(gr, corners_r, (11, 11), (-1, -1), criteria)
                obj_points.append(obj_pts_template.copy())
                pts_l.append(corners_l)
                pts_r.append(corners_r)
                n += 1
                print(f"  Captured pair {n}")
            else:
                print("  Checkerboard not found in both frames — try again.")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()

    if n < 5:
        print(f"Only {n} pairs captured — need at least 5. Aborting.")
        sys.exit(1)

    print(f"\nCalibrating with {n} stereo pairs…")
    image_size = (args.width, args.height)

    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, pts_l, image_size, None, None)
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, pts_r, image_size, None, None)
    print(f"  Left  RMS: {ret_l:.4f}")
    print(f"  Right RMS: {ret_r:.4f}")

    flags = (cv2.CALIB_FIX_INTRINSIC)
    ret_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, pts_l, pts_r, K_l, D_l, K_r, D_r, image_size,
        criteria=criteria, flags=flags
    )
    print(f"  Stereo RMS: {ret_s:.4f}")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T, alpha=0
    )

    np.savez(args.output,
             K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             image_size=np.array(image_size))
    print(f"\nCalibration saved to {args.output}")
    print(f"Baseline: {np.linalg.norm(T)*1000:.1f} mm")
    print(f"Focal length (left): {K_l[0,0]:.1f} px")


if __name__ == "__main__":
    main()

"""Stereo mount alignment assistant (M2, before any calibration).

Two cameras on adjustable mounts must point the same way before they are
calibrated: relative pitch breaks row matching, roll tilts the rows, and yaw
eats the overlap and the disparity range. This module measures how each camera
is rotated, from feature matches on a distant textured scene, and turns the
numbers into spoken adjustments.

Rotation convention (all angles in degrees, for the camera being adjusted,
relative to a reference camera, as felt by the wearer standing behind it):

    yaw   > 0  camera points further RIGHT than the reference
    pitch > 0  camera points further UP
    roll  > 0  camera is turned CLOCKWISE

The left-to-right pair rotation is measured directly. Each stereo camera is
also measured against the wide camera, which is fixed to the frame and serves
as "straight ahead", so the advice can name which camera to move. Wide/stereo
focal ratio is estimated from the homography scale, so only the stereo focal
length needs a provisional value.

Parallax: at finite distance a baseline B shifts the right image left by
f*B/Z, which reads as extra right-camera toe-out of atan(B/Z). That term is
subtracted using the configured scene distance; aim at the farthest view
available (>= 4 m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Rotation helpers (convention above)
# ---------------------------------------------------------------------------
def rotation_from_angles(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R such that a ray d_ref in the reference camera is d = R @ d_ref in the
    adjusted camera, for a camera turned by (yaw, pitch, roll) degrees."""
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), math.sin(r), 0], [-math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def angles_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rotation_from_angles. The bottom row of Rz @ Rx @ Ry equals
    the bottom row of Rx @ Ry = [cos p sin y, -sin p, cos p cos y], and
    R[0,1] / R[1,1] = sin r cos p / cos r cos p."""
    pitch = math.degrees(math.asin(float(np.clip(-R[2, 1], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(R[2, 0]), float(R[2, 2])))
    roll = math.degrees(math.atan2(float(R[0, 1]), float(R[1, 1])))
    return yaw, pitch, roll


def intrinsics(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    return R


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
@dataclass
class Match:
    pts_a: np.ndarray        # inlier points in image a (N, 2)
    pts_b: np.ndarray        # inlier points in image b (N, 2)
    H: np.ndarray            # homography a -> b
    n_matches: int


_ORB = None


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def match_images(a: np.ndarray, b: np.ndarray, n_features: int = 2500,
                 min_inliers: int = 40) -> Match | None:
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(n_features, fastThreshold=10)
    ga, gb = _gray(a), _gray(b)
    ka, da = _ORB.detectAndCompute(ga, None)
    kb, db = _ORB.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < min_inliers or len(kb) < min_inliers:
        return None
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in knn if rest and m.distance < 0.8 * rest[0].distance]
    if len(good) < min_inliers:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_inliers:
        return None
    return Match(pa[inl], pb[inl], H, len(good))


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
@dataclass
class Rotation:
    yaw: float
    pitch: float
    roll: float
    inliers: int = 0
    scale: float = 1.0        # focal ratio f_b / f_a recovered from H


def rotation_from_homography(H: np.ndarray, K_a: np.ndarray, K_b: np.ndarray,
                             estimate_scale: bool = False) -> Rotation:
    """Pure-rotation model H ~ K_b R K_a^-1. With estimate_scale, K_b's focal
    is rescaled by the factor that makes K_b^-1 H K_a closest to a rotation."""
    s = 1.0
    if estimate_scale:
        A = H[:2, :2] / H[2, 2]
        s = math.sqrt(abs(np.linalg.det(A))) or 1.0
        K_b = K_b.copy()
        K_b[0, 0] = K_a[0, 0] * s
        K_b[1, 1] = K_a[1, 1] * s
    M = np.linalg.inv(K_b) @ H @ K_a
    M /= np.cbrt(np.linalg.det(M))
    R = nearest_rotation(M)
    yaw, pitch, roll = angles_from_rotation(R)
    return Rotation(yaw, pitch, roll, scale=s)


@dataclass
class PairMetrics:
    vertical_px: float        # median (y_right - y_left)
    roll_slope_deg: float     # tilt of dy against x
    disparity_px: float       # median (x_left - x_right)
    overlap: float            # fraction of the left image that lands in the right
    inliers: int


def pair_metrics(m: Match, shape: tuple[int, int]) -> PairMetrics:
    h, w = shape
    dy = m.pts_b[:, 1] - m.pts_a[:, 1]
    dx = m.pts_a[:, 0] - m.pts_b[:, 0]
    A = np.vstack([m.pts_a[:, 0] - w / 2, np.ones(len(dy))]).T
    slope = float(np.linalg.lstsq(A, dy, rcond=None)[0][0])
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, m.H).reshape(-1, 2)
    poly = cv2.convexHull(proj.astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(poly, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return PairMetrics(float(np.median(dy)), math.degrees(math.atan(slope)),
                       float(np.median(dx)), float(inter) / (w * h), len(dy))


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
@dataclass
class Targets:
    """Pre-calibration tolerances for hand-adjusted mounts (2026-10-21).

    3 px / 0.5 deg / 1 deg could not be held by hand: tightening a knob moved
    the camera past them, and the roll estimate itself scatters about
    +/-0.8 deg, so the session never finished. These values are well inside
    what rectification corrects exactly; the calibration gate (held-out row
    error < 1 px) is the real acceptance test."""
    vertical_px: float = 8.0
    roll_deg: float = 1.0
    yaw_deg: float = 2.0
    pitch_deg: float = 0.5

    def scaled(self, k: float) -> "Targets":
        return Targets(self.vertical_px * k, self.roll_deg * k, self.yaw_deg * k, self.pitch_deg * k)


@dataclass
class Assessment:
    pair: Rotation | None = None             # right relative to left (parallax removed)
    metrics: PairMetrics | None = None
    left_vs_wide: Rotation | None = None
    right_vs_wide: Rotation | None = None
    advice: str = ""
    aligned: bool = False
    detail: dict = field(default_factory=dict)


def _amount(deg: float) -> str:
    a = abs(deg)
    if a < 1.5:
        return "very slightly"
    if a < 4:
        return "a little"
    return f"about {int(round(a))} degrees"


def _yaw_words(cam: str, err_yaw: float) -> str:
    """Correction for a camera pointing err_yaw degrees too far right."""
    turn_left = err_yaw > 0
    inward = (cam == "left") != turn_left   # left cam turning right = inward
    return "inward" if inward else "outward"


def advise(pair: Rotation, metrics: PairMetrics, left_w: Rotation | None,
           right_w: Rotation | None, t: Targets = Targets(),
           f_px: float = 550.0, sticky: dict | None = None) -> tuple[str, bool]:
    """One spoken instruction, worst error first (normalised by tolerance).

    The pair rotation decides *whether* the cameras are parallel. With the
    wide camera as "straight ahead", the camera that deviates more from it is
    the one to move, by its own error; without it the right camera is moved
    by the pair error."""
    errs = {"yaw": abs(pair.yaw) / t.yaw_deg,
            "pitch": abs(metrics.vertical_px) / t.vertical_px,
            "roll": abs(pair.roll) / t.roll_deg}
    if sticky is not None:
        for kind in [k for k in sticky if errs.get(k, 0) <= 1.0]:
            del sticky[kind]
    if max(errs.values()) <= 1.0:
        return "Aligned. Lock the mounts now.", True
    worst = max(errs, key=errs.get)

    have_ref = left_w is not None and right_w is not None
    if have_ref:
        cam = "left" if abs(getattr(left_w, worst)) > abs(getattr(right_w, worst)) else "right"
        if sticky is not None:
            # Once a camera is named for an error, keep naming it until that
            # error is fixed: a noisy reference flipped "right" to "left"
            # mid-adjustment on 2026-10-21.
            cam = sticky.setdefault(worst, cam)
        err = getattr(left_w if cam == "left" else right_w, worst)
        # Never move a camera more than the pair needs: cap at the pair error.
        # Pitch is judged by vertical pixels, not pair.pitch, so it is not capped.
        pair_err = getattr(pair, worst)
        if worst != "pitch" and abs(err) > abs(pair_err):
            err = math.copysign(abs(pair_err), err)
    else:
        cam, err = "right", getattr(pair, worst)

    if worst == "yaw":
        return f"Turn the {cam} camera {_yaw_words(cam, err)}, {_amount(err)}.", False
    if worst == "pitch":
        # The size of a pitch error comes from the vertical pixel offset, which
        # holds at any scene depth; the homography pitch does not when a near
        # object dominates (70 px offset read as 1.4 deg on 2026-10-21).
        size = math.degrees(math.atan(abs(metrics.vertical_px) / f_px))
        if have_ref:
            sign = 1.0 if err > 0 else -1.0
        else:
            # right content lower (dy > 0) means the right camera points up
            sign = 1.0 if metrics.vertical_px > 0 else -1.0
        err = sign * size
        direction = "down" if err > 0 else "up"
        # Live countdown in 5 px steps: the sentence changes as the knob turns,
        # so the person hears progress instead of one repeated instruction.
        left_px = int(round(abs(metrics.vertical_px) / 5.0) * 5)
        return (f"Tilt the {cam} camera {direction}, {_amount(err)}. "
                f"{left_px} pixels to go."), False
    direction = "anticlockwise" if err > 0 else "clockwise"
    return (f"Rotate the {cam} camera {direction}, as you look at it from behind, "
            f"{_amount(err)}."), False


def assess(left: np.ndarray, right: np.ndarray, wide: np.ndarray | None,
           f_stereo: float, baseline_m: float = 0.1, scene_m: float = 4.0,
           targets: Targets = Targets()) -> Assessment:
    h, w = left.shape[:2]
    K_s = intrinsics(f_stereo, w, h)
    out = Assessment()
    m = match_images(left, right)
    if m is None:
        out.advice = "I cannot match the two stereo cameras. Face a detailed scene far away."
        return out
    pair = rotation_from_homography(m.H, K_s, K_s)
    pair.inliers = len(m.pts_a)
    pair.yaw -= math.degrees(math.atan2(baseline_m, scene_m))  # parallax reads as toe-out
    out.pair, out.metrics = pair, pair_metrics(m, (h, w))
    if wide is not None:
        ws = cv2.resize(wide, (w, int(round(wide.shape[0] * w / wide.shape[1]))),
                        interpolation=cv2.INTER_AREA)
        K_w = intrinsics(f_stereo, ws.shape[1], ws.shape[0])
        for name, img in (("left_vs_wide", left), ("right_vs_wide", right)):
            mw = match_images(ws, img)
            if mw is not None:
                r = rotation_from_homography(mw.H, K_w, K_s, estimate_scale=False)
                r.inliers = len(mw.pts_a)
                setattr(out, name, r)
    out.advice, out.aligned = advise(pair, out.metrics, out.left_vs_wide, out.right_vs_wide, targets,
                                     f_stereo)
    return out


# ---------------------------------------------------------------------------
# Live guided session
# ---------------------------------------------------------------------------
def _median_rotation(items: list[Rotation]) -> Rotation:
    return Rotation(float(np.median([r.yaw for r in items])),
                    float(np.median([r.pitch for r in items])),
                    float(np.median([r.roll for r in items])),
                    int(np.median([r.inliers for r in items])))


def _median_metrics(items: list[PairMetrics]) -> PairMetrics:
    return PairMetrics(*(float(np.median([getattr(m, k) for m in items]))
                         for k in ("vertical_px", "roll_slope_deg", "disparity_px", "overlap")),
                       int(np.median([m.inliers for m in items])))


def _render(left, right, a: Assessment, aligned_for: float, hold_s: float) -> np.ndarray:
    """Side-by-side view with horizontal guide lines: a matching feature must
    sit on the same line in both images when pitch and roll are right."""
    pair = np.hstack([left, right])
    h, w = pair.shape[:2]
    for y in range(0, h, 50):
        cv2.line(pair, (0, y), (w, y), (0, 200, 255), 1)
    cv2.line(pair, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    canvas = np.zeros((1080, 1920, 3), np.uint8)
    s = min(1900 / w, 760 / h)
    view = cv2.resize(pair, (int(w * s), int(h * s)))
    y0 = 1080 - view.shape[0] - 10
    canvas[y0:y0 + view.shape[0], 10:10 + view.shape[1]] = view
    if a.pair is not None and a.metrics is not None:
        p, m = a.pair, a.metrics
        txt = (f"yaw {p.yaw:+.1f} deg   vertical {m.vertical_px:+.0f} px   roll {p.roll:+.2f} deg   "
               f"overlap {m.overlap:.0%}   matches {p.inliers}")
        cv2.putText(canvas, txt, (20, y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (80, 255, 80) if a.aligned else (255, 255, 255), 2, cv2.LINE_AA)
    if aligned_for > 0:
        cv2.putText(canvas, f"holding aligned {aligned_for:.0f} / {hold_s:.0f} s", (20, y0 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 80), 2, cv2.LINE_AA)
    return canvas


def run_align(cfg, scene_m: float = 4.0, baseline_m: float = 0.13, f_stereo: float = 550.0,
              hold_s: float = 10.0, max_s: float = 900.0, speak: bool = True,
              targets: Targets = Targets()) -> dict:
    """Guide the human until the pair is aligned and stays aligned for hold_s
    seconds (long enough to tighten the mounts without disturbing them)."""
    import json
    import time
    from pathlib import Path
    from .doctor import _open_probes, _wait_frames, assign_roles, capture_nodes
    from .guide import Guide

    guide = Guide(cfg.speech, "ARGUS align", speak=speak, show=True, min_gap_s=2.0, repeat_s=10.0)
    roles = assign_roles(capture_nodes(), cfg)
    if "left" not in roles or "right" not in roles:
        guide.failed("Alignment needs both stereo cameras plugged in.")
        guide.close()
        return {"error": "need both stereo cameras"}
    probes = _open_probes(cfg, roles, wide="wide" in roles)
    history: list[Assessment] = []
    log = []
    report: dict = {}
    try:
        _wait_frames(probes)
        guide.announce(f"Alignment. Face the farthest view you can, at least {scene_m:.0f} metres "
                       "away, with plenty of detail. Do not point the cameras at this monitor: "
                       "it shows their own pictures and confuses the measurement. "
                       "Hold the rig still between adjustments.")
        t0 = time.monotonic()
        aligned_since = None
        locked_announced = False
        sticky: dict[str, str] = {}
        last_print = 0.0
        while time.monotonic() - t0 < max_s:
            L, R = probes["left"].snapshot(1.0), probes["right"].snapshot(1.0)
            W = probes["wide"].snapshot(1.0) if "wide" in probes else None
            if L is None or R is None:
                frozen = [r for r, f in (("left", L), ("right", R)) if f is None
                          and probes[r].seq > 0]
                if frozen:
                    guide.say(f"The {frozen[0]} camera froze. Check its cable. "
                              "I will reconnect it automatically.")
                    guide.show(sub="Waiting for the camera to come back")
                time.sleep(0.1)
                continue
            a = assess(L, R, W, f_stereo, baseline_m, scene_m, targets)
            if a.pair is not None:
                history.append(a)
                history[:] = history[-5:]
                lw = [h.left_vs_wide for h in history if h.left_vs_wide is not None]
                rw = [h.right_vs_wide for h in history if h.right_vs_wide is not None]
                sm = Assessment(pair=_median_rotation([h.pair for h in history]),
                                metrics=_median_metrics([h.metrics for h in history]),
                                left_vs_wide=_median_rotation(lw) if len(lw) >= 3 else None,
                                right_vs_wide=_median_rotation(rw) if len(rw) >= 3 else None)
                # Hysteresis: once aligned, only a clear excursion (1.5x the
                # tolerance) counts as "moved"; noise alone must not send the
                # person back to the knobs while they lock the mounts.
                active = targets.scaled(1.5) if aligned_since is not None else targets
                sm.advice, sm.aligned = advise(sm.pair, sm.metrics, sm.left_vs_wide,
                                               sm.right_vs_wide, active, f_stereo, sticky)
                a = sm
                if time.monotonic() - last_print >= 1.0:
                    last_print = time.monotonic()
                    print(f"[measure] yaw {a.pair.yaw:+.1f} vertical {a.metrics.vertical_px:+.0f}px "
                          f"roll {a.pair.roll:+.1f} overlap {a.metrics.overlap:.0%} "
                          f"matches {a.pair.inliers}", flush=True)
                log.append({"t": round(time.monotonic() - t0, 2), "yaw": a.pair.yaw,
                            "pitch": a.pair.pitch, "roll": a.pair.roll,
                            "vertical_px": a.metrics.vertical_px, "overlap": a.metrics.overlap,
                            "inliers": a.pair.inliers, "advice": a.advice})
            else:
                history.clear()
            now = time.monotonic()
            if a.aligned:
                aligned_since = aligned_since or now
                if not locked_announced:
                    guide.say("Aligned. Lock the mounts now, gently. Keep the rig still.", force=True)
                    locked_announced = True
                else:
                    guide.say(f"Still aligned. {int(hold_s - (now - aligned_since)) + 1} seconds.")
                if now - aligned_since >= hold_s:
                    p, m = a.pair, a.metrics
                    report = {"kind": "argus-align", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "result": "aligned", "scene_m": scene_m, "baseline_m": baseline_m,
                              "f_stereo_px": f_stereo, "targets": targets.__dict__,
                              "final": {"yaw_deg": p.yaw, "pitch_deg": p.pitch, "roll_deg": p.roll,
                                        "vertical_px": m.vertical_px, "overlap": m.overlap,
                                        "inliers": p.inliers},
                              "duration_s": round(now - t0, 1), "trace": log[-200:]}
                    guide.done("Mounts aligned and held still. You can start calibration.")
                    break
            else:
                if aligned_since is not None and locked_announced:
                    guide.say("It moved while locking. " + a.advice, force=True)
                aligned_since = None
                locked_announced = False
                guide.say(a.advice)
            key = guide.show(_render(L, R, a, (now - aligned_since) if aligned_since else 0, hold_s),
                             sub="Lines must pass through the same features in both pictures. Esc to stop.")
            # Esc only: a stray "q" typed elsewhere while this window has focus
            # ended the first live session after 2 s (2026-10-18).
            if key == 27:
                guide.failed("Stopped before alignment finished.")
                report = {"kind": "argus-align", "result": "stopped", "trace": log[-200:]}
                break
        else:
            guide.failed("Out of time before the cameras were aligned.")
            report = {"kind": "argus-align", "result": "timeout", "trace": log[-200:]}
        if report:
            out = Path(__file__).resolve().parents[1] / "reports" / \
                f"align-{time.strftime('%Y-%m-%d-%H%M')}.json"
            out.write_text(json.dumps(report, indent=2, default=float))
            report["path"] = str(out)
        return report
    finally:
        for p in probes.values():
            p.close()
        guide.close()

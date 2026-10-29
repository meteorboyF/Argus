"""M2 calibration quality gate and held-out row error on a synthetic rig."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_stereo", Path(__file__).parents[1] / "scripts" / "calibrate_stereo.py")
cs = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cs  # @dataclass needs the module registered
_SPEC.loader.exec_module(cs)

SIZE = (960, 600)
K = np.array([[550.0, 0, 480], [0, 550.0, 300], [0, 0, 1]])
D = np.zeros(5)


def _rig_views(n=20, baseline=0.13, noise=0.0, seed=0):
    """7x7 inner-corner board (25 mm squares) seen by an ideal parallel pair."""
    rng = np.random.default_rng(seed)
    obj = np.zeros((49, 3), np.float32)
    obj[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 0.025
    objs, pl, pr = [], [], []
    T_lr = np.array([-baseline, 0, 0])
    while len(objs) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.25, 0.15), rng.uniform(-0.2, 0.1), rng.uniform(0.6, 1.6)])
        il, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        ir, _ = cv2.projectPoints(obj, rvec, tvec + T_lr, K, D)
        il, ir = il.reshape(-1, 2), ir.reshape(-1, 2)
        if (il.min() < 5 or ir.min() < 5 or il[:, 0].max() > SIZE[0] - 5
                or ir[:, 0].max() > SIZE[0] - 5 or il[:, 1].max() > SIZE[1] - 5
                or ir[:, 1].max() > SIZE[1] - 5):
            continue
        il += rng.normal(0, noise, il.shape)
        ir += rng.normal(0, noise, ir.shape)
        objs.append(obj.copy())
        pl.append(il.reshape(-1, 1, 2).astype(np.float32))
        pr.append(ir.reshape(-1, 1, 2).astype(np.float32))
    return objs, pl, pr


def test_clean_rig_has_small_held_out_row_error():
    objs, pl, pr = _rig_views(noise=0.1)
    mean, mx, n = cs.held_out_row_error(objs, pl, pr, SIZE)
    assert n == 4
    assert mean < 0.5 and mx < 1.0


def test_joint_corner_order_recovers_known_rig_across_diagonal_switch():
    """Project a real 101 mm rig with relative roll; fit after detector ordering."""
    rng = np.random.default_rng(42)
    obj = cs.make_object_points((7,7), .024)
    rotation, _ = cv2.Rodrigues(np.array([.01,.03,np.deg2rad(2)]))
    translation = np.array([-.101, .001, .002])
    objs, pl, pr = [], [], []
    for i in range(24):
        rv = np.array([rng.uniform(-.5,.5),rng.uniform(-.5,.5),
                       np.deg2rad(rng.uniform(40,48))])
        tv = np.array([rng.uniform(-.18,.1),rng.uniform(-.14,.08),rng.uniform(.55,1.0)])
        rb, _ = cv2.Rodrigues(rv)
        rr, _ = cv2.Rodrigues(rotation@rb)
        il, _ = cv2.projectPoints(obj,rv,tv,K,D)
        ir, _ = cv2.projectPoints(obj,rr,rotation@tv+translation,K,D)
        il = cs.normalize_corners(il,(7,7))
        ir = cs.normalize_corners(ir,(7,7))
        matched = cs.align_stereo_corners(il,ir,(7,7))
        assert matched is not None
        objs.append(obj.copy()); pl.append(il); pr.append(matched)
    rms, _, _, _, _, recovered_rotation, recovered_translation = cs.calibrate(objs,pl,pr,SIZE)
    assert rms < .01
    np.testing.assert_allclose(recovered_translation.ravel(),translation,atol=.001)
    np.testing.assert_allclose(recovered_rotation,rotation,atol=.001)
    mean, maximum, held = cs.held_out_row_error(objs,pl,pr,SIZE)
    assert held >= 3 and mean < .1 and maximum < .2


def test_gate_accepts_good_calibration():
    why = cs.quality_gate(mono_rms={"left": 0.3, "right": 0.35}, stereo_rms=0.45, row_mean=0.2,
                          row_max=0.7, baseline_m=0.131, tape_baseline_m=0.13, n_views=18,
                          coverage=7 / 9)
    assert why == []


def test_gate_names_every_failure():
    why = cs.quality_gate(mono_rms={"left": 0.9, "right": 0.3}, stereo_rms=1.2, row_mean=0.6,
                          row_max=1.4, baseline_m=0.16, tape_baseline_m=0.13, n_views=12,
                          coverage=0.5)
    text = " | ".join(why)
    for needle in ("left camera", "stereo error", "held-out row error", "worst held-out",
                   "baseline", "only 12 views", "covered 50%"):
        assert needle in text
    assert "right camera" not in text


def test_gate_requires_tape_baseline():
    why = cs.quality_gate(mono_rms={"left": 0.3, "right": 0.3}, stereo_rms=0.4, row_mean=0.2,
                          row_max=0.5, baseline_m=0.13, tape_baseline_m=None, n_views=20,
                          coverage=1.0)
    assert why == ["baseline not checked against a tape measurement"]


def test_gate_rejects_unstable_intrinsics_and_flat_only_views():
    why = cs.quality_gate(mono_rms={'left':.3,'right':.3},stereo_rms=.4,
                          row_mean=.2,row_max=.5,baseline_m=.101,tape_baseline_m=.101,
                          n_views=20,coverage=1.,intrinsic_fit={'left':{'stable':False}},
                          perspective_spans=[.01,.12])
    assert any('intrinsics are unstable' in reason for reason in why)
    assert any('perspective variation' in reason for reason in why)


def test_gate_rejects_nan():
    why = cs.quality_gate(mono_rms={"left": float("nan")}, stereo_rms=float("nan"),
                          row_mean=float("nan"), row_max=float("nan"), baseline_m=0.13,
                          tape_baseline_m=0.13, n_views=20, coverage=1.0)
    assert len(why) == 4

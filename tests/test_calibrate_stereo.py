import importlib.util
from collections import deque
import sys
from pathlib import Path

import numpy as np
import time
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_stereo", Path(__file__).parents[1] / "scripts" / "calibrate_stereo.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE  # @dataclass needs the module registered
_SPEC.loader.exec_module(_MODULE)
is_diverse_pose = _MODULE.is_diverse_pose
pose_signature = _MODULE.pose_signature
ThreadedStereoCapture = _MODULE.ThreadedStereoCapture
movement_instruction = _MODULE.movement_instruction


def _corners(offset_x=0, offset_y=0, scale=1.0):
    pts = np.mgrid[0:9, 0:6].T.reshape(-1, 2).astype(np.float32)
    pts = pts * (20 * scale) + np.array([offset_x, offset_y], dtype=np.float32)
    return pts.reshape(-1, 1, 2)


def test_near_duplicate_calibration_pose_is_rejected():
    first = pose_signature(_corners(100, 80), 600, 960)
    repeat = pose_signature(_corners(101, 81), 600, 960)
    assert not is_diverse_pose(repeat, [first])


def test_translated_or_scaled_calibration_pose_is_accepted():
    first = pose_signature(_corners(100, 80), 600, 960)
    translated = pose_signature(_corners(250, 300), 600, 960)
    scaled = pose_signature(_corners(100, 80, 1.8), 600, 960)
    assert is_diverse_pose(translated, [first])
    assert is_diverse_pose(scaled, [first])


def test_threaded_capture_does_not_serialize_blocking_cameras():
    class FakeCamera:
        def __init__(self, value, delay):
            self.value, self.delay, self.released = value, delay, False

        def read(self):
            time.sleep(self.delay)
            return (False, None) if self.released else (
                True, np.full((2, 2, 3), self.value, dtype=np.uint8))

        def release(self):
            self.released = True

    capture = ThreadedStereoCapture(FakeCamera(1, 0.01), FakeCamera(2, 0.02))
    deadline = time.monotonic() + 0.2
    pair = None
    while pair is None and time.monotonic() < deadline:
        pair = capture.pair()
        time.sleep(0.005)
    capture.close()
    assert pair is not None
    assert pair[0][0, 0, 0] == 1
    assert pair[1][0, 0, 0] == 2


def test_calibration_instructions_forbid_hinge_changes():
    assert "WHOLE rig" in movement_instruction(0)
    assert "hinges locked" in movement_instruction(14)


def test_new_scale_in_same_coverage_cell_can_capture():
    first = _corners(200, 200)
    changed = _corners(168, 180, 1.4)
    assert _MODULE._coverage_cell(first, 960, 600) == _MODULE._coverage_cell(changed, 960, 600)
    diverse = is_diverse_pose(pose_signature(changed, 960, 600),
                              [pose_signature(first, 960, 600)])
    assert diverse
    assert _MODULE.capture_readiness(both=True, steady=True, diverse=diverse,
                                     span_frac=.3, since_capture=2) == 'READY'


@pytest.mark.parametrize('overrides,status,heading', [
    ({'steady':False}, 'HOLD STILL', 'HOLD STILL'),
    ({'diverse':False,'steady':False}, 'MOVE/TILT', 'SAME POSE AS BEFORE'),
    ({'since_capture':.5}, 'WAIT', 'WAIT FOR NEXT CAPTURE'),
    ({'span_frac':.15}, 'RESIZE', 'BRING THE BOARD CLOSER'),
    ({'span_frac':.9}, 'RESIZE', 'MOVE THE BOARD FARTHER AWAY'),
    ({}, 'READY', 'GOOD, HOLD STILL'),
])
def test_coach_explains_the_capture_blocker(overrides, status, heading):
    state = dict(both=True, steady=True, diverse=True, span_frac=.3, since_capture=2)
    state.update(overrides)
    assert _MODULE.capture_readiness(**state) == status
    coach = _MODULE.Coach(15)
    head, _ = coach.advise(pattern=(7,7), okcL=True, okcR=True,
                           corners=_corners(200,200), size=(960,600),
                           n_views=2, covered={4},
                           **{k:v for k,v in state.items() if k != 'both'})
    assert head == heading


def test_diagonal_board_correspondence_uses_joint_order():
    # A 2-degree camera roll straddles the old per-image 45-degree switch.
    points = np.mgrid[0:7,0:7].T.reshape(-1,2).astype(np.float32)*20
    def rotate(degrees, offset):
        a = np.deg2rad(degrees)
        r = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        return (points@r.T+offset).astype(np.float32).reshape(-1,1,2)
    raw_l, raw_r = rotate(44, [350,150]), rotate(46, [290,155])
    left = _MODULE.normalize_corners(raw_l,(7,7))
    right = _MODULE.normalize_corners(raw_r,(7,7))
    indices = np.argmin(np.linalg.norm(left[:,0,None,:]-raw_l[None,:,0,:],axis=-1),axis=1)
    expected = raw_r[indices]
    assert not np.allclose(right,expected)  # reproduces previous bug
    aligned = _MODULE.align_stereo_corners(left,right,(7,7))
    np.testing.assert_allclose(aligned,expected)


def test_ambiguous_or_invalid_corner_alignment_is_rejected():
    points = np.mgrid[0:7,0:7].T.reshape(-1,2).astype(np.float32)*20
    a = np.deg2rad(45)
    r = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    left = points.reshape(-1,1,2)
    right = (points@r.T).astype(np.float32).reshape(-1,1,2)
    assert _MODULE.align_stereo_corners(left,right,(7,7)) is None
    assert _MODULE.align_stereo_corners(left,np.zeros_like(left),(7,7)) is None


def _manual_capture(left_history, right_history, consumed=(-1,-1)):
    capture = ThreadedStereoCapture.__new__(ThreadedStereoCapture)
    capture._lock = __import__('threading').Lock()
    capture._history = [deque(left_history),deque(right_history)]
    capture._last_returned = consumed
    capture.max_skew_ms = 12
    return capture


def test_calibration_pair_requires_both_sequences_to_advance():
    frame = np.ones((2,2,3),np.uint8)
    now = time.monotonic()
    capture = _manual_capture([(frame,now,2)],[(frame,now,1)],(1,1))
    assert capture.pair() is None


def test_calibration_pair_rejects_stale_and_skewed_frames():
    frame = np.ones((2,2,3),np.uint8)
    now = time.monotonic()
    for lt,rt in ((now-1,now-1),(now-.04,now)):
        capture = _manual_capture([(frame,lt,1)],[(frame,rt,1)])
        assert capture.pair() is None


def test_calibration_pairs_compatible_history_and_returns_owned_images():
    frame = np.ones((2,2,3),np.uint8)
    now = time.monotonic()
    capture = _manual_capture([(frame,now-.02,1),(frame,now,2)],
                              [(frame,now-.021,1)])
    left,right,skew = capture.pair()
    assert capture._last_returned == (1,1) and skew < 2
    left[:] = right[:] = 0
    assert frame.any()


def test_translation_scale_and_flat_rotation_do_not_count_as_tilts():
    points = _corners(200,200).reshape(-1,2)
    a = np.deg2rad(30)
    r = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    original = _MODULE.tilt_signature(points.reshape(-1,1,2),(9,6))
    transformed = _MODULE.tilt_signature((points@r.T*1.5+20).reshape(-1,1,2),(9,6))
    np.testing.assert_allclose(original,transformed,atol=1e-6)
    assert _MODULE.tilt_instruction([original,transformed])[0] == 'TILT TOP THEN BOTTOM'


def test_tilt_guidance_requires_both_perspective_directions():
    assert _MODULE.tilt_instruction([[0,0],[.15,0]])[0] == 'TILT LEFT THEN RIGHT'
    assert _MODULE.tilt_instruction([[0,0],[.15,.15]]) is None


def test_multistart_optimizes_guesses_and_selects_best_fit(monkeypatch):
    calls = []
    def fit(objects,points,size,guess,dist,**kwargs):
        calls.append((guess.copy(),kwargs['flags']))
        k = np.array([[650.,0,480],[0,650.,300],[0,0,1]])
        return (.9 if len(calls)==1 else .3),k,np.zeros(5),[],[]
    monkeypatch.setattr(_MODULE.cv2,'calibrateCamera',fit)
    best,info = _MODULE.fit_intrinsics([],[],(960,600))
    assert best[0] == .3 and info['stable']
    np.testing.assert_allclose([c[0][0,0] for c in calls],[600,800,1000])
    assert all(flags & _MODULE.cv2.CALIB_USE_INTRINSIC_GUESS for _,flags in calls)
    assert best[1][0,0] == 650  # fitted value is not a fixed input guess


def test_multistart_flags_equal_error_but_inconsistent_focal_lengths(monkeypatch):
    def fit(objects,points,size,guess,dist,**kwargs):
        return .3,guess,np.zeros(5),[],[]
    monkeypatch.setattr(_MODULE.cv2,'calibrateCamera',fit)
    _,info = _MODULE.fit_intrinsics([],[],(960,600))
    assert not info['stable']
    assert info['focal_spread_fraction'] > .1


def test_sector_detector_finds_synthetic_board_without_classic_refinement(monkeypatch):
    image=np.full((500,600),230,np.uint8)
    for row in range(8):
        for col in range(8):
            image[80+row*40:120+row*40,100+col*40:140+col*40] = 0 if (row+col)%2==0 else 255
    def forbidden(*args,**kwargs):
        raise AssertionError('Sector coordinates must not undergo classic refinement')
    monkeypatch.setattr(_MODULE.cv2,'cornerSubPix',forbidden)
    ok,points=_MODULE.detect_board(image,(7,7),'sector')
    assert ok and points.shape==(49,1,2)


def test_sector_detector_rejects_blank_image():
    ok,points=_MODULE.detect_board(np.full((200,200),128,np.uint8),(7,7),'sector')
    assert not ok and points is None

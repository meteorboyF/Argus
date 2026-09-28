import importlib.util
from pathlib import Path

import numpy as np
import time

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_stereo", Path(__file__).parents[1] / "scripts" / "calibrate_stereo.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
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

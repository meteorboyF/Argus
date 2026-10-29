"""Reading must not steal stereo pairs or restart a safety camera."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from argus.cameras import CameraRig
from argus.config import CameraConfig


@pytest.mark.parametrize("source", ["left", "right", "wide"])
def test_reading_uses_selected_view_without_consuming_stereo(source):
    rig = CameraRig.__new__(CameraRig)
    rig.cfg = CameraConfig(reading_source=source)
    rig._last_pair = (7, 9)
    frames = {}
    for i, name in enumerate(("left", "right", "wide")):
        frames[name] = np.full((4, 6, 3), i + 1, np.uint8)
        setattr(rig, name, SimpleNamespace(
            latest=lambda name=name: (frames[name], time.perf_counter(), 10)))
    frame = rig.get_reading_frame()
    np.testing.assert_array_equal(frame, frames[source])
    frame[:] = 0
    assert frames[source].any()
    assert rig._last_pair == (7, 9)
    if source != "wide":
        # No _Reader or capture object is available: any reopen would fail.
        np.testing.assert_array_equal(rig.capture_reading_still(), frames[source])


@pytest.mark.parametrize("age", [1.0, -1.0])
def test_reading_rejects_stale_or_future_stereo_frames(age):
    rig = CameraRig.__new__(CameraRig)
    rig.cfg = CameraConfig(reading_source="left")
    rig.left = SimpleNamespace(latest=lambda: (
        np.ones((4, 6, 3), np.uint8), time.perf_counter() - age, 1))
    assert rig.capture_reading_still() is None


def test_invalid_source_fails_before_opening_cameras():
    with pytest.raises(ValueError, match="reading_source"):
        CameraRig(CameraConfig(reading_source="both"))


def test_wide_reading_retains_still_capture():
    rig = CameraRig.__new__(CameraRig)
    rig.cfg = CameraConfig(reading_source="wide")
    still = np.ones((8, 8, 3), np.uint8)
    rig.capture_still = lambda: still
    assert rig.capture_reading_still() is still

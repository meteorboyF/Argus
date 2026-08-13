import cv2
import numpy as np
import pytest

from argus.cameras import transform_frame, transformed_size


def test_opposite_mount_rotations_produce_matching_upright_frames():
    upright = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
    upright = cv2.cvtColor(upright, cv2.COLOR_GRAY2BGR)
    raw_left = cv2.rotate(upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raw_right = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)

    assert np.array_equal(transform_frame(raw_left, 90), upright)
    assert np.array_equal(transform_frame(raw_right, 270), upright)


def test_transform_flips_after_rotation():
    frame = np.arange(2 * 3, dtype=np.uint8).reshape(2, 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    expected = cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 1)
    assert np.array_equal(transform_frame(frame, 90, flip_horizontal=True), expected)


@pytest.mark.parametrize("rotation,size", [
    (0, (960, 600)), (90, (600, 960)), (180, (960, 600)), (270, (600, 960)),
])
def test_transformed_size(rotation, size):
    assert transformed_size(960, 600, rotation) == size


def test_invalid_rotation_rejected():
    with pytest.raises(ValueError, match="0/90/180/270"):
        transform_frame(np.zeros((2, 2, 3), dtype=np.uint8), 45)

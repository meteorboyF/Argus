import cv2
import numpy as np
import pytest
import threading

from argus.config import DepthConfig
from argus.depth import DepthEstimator


class FakeMatcher:
    def __init__(self, result):
        self.result = result
        self.call = None

    def compute(self, left, right, **kwargs):
        self.call = (left.shape, right.shape, kwargs)
        return self.result.copy()


def test_cuda_disparity_restores_full_resolution_units():
    estimator = DepthEstimator.__new__(DepthEstimator)
    estimator.cfg = DepthConfig(fast_downscale=2, num_disparities=128,
                                block_size=5, sad_uniqueness_percent=5)
    matcher = FakeMatcher(np.full((30, 40), 7, dtype=np.float32))
    estimator._cuda_matcher = matcher
    frame = np.zeros((60, 80, 3), dtype=np.uint8)

    disparity = estimator._cuda_disparity(frame, frame)

    assert disparity.shape == (60, 80)
    assert np.all(disparity == 14)
    assert matcher.call == ((30, 40), (30, 40), {
        "max_disparity": 64, "radius": 2, "uniqueness_percent": 5})


def test_cuda_matcher_recovers_synthetic_shift():
    pytest.importorskip("pycuda")
    if not cv2.cuda.getCudaEnabledDeviceCount() and not __import__("pathlib").Path(
            "/usr/local/cuda/bin/nvcc").is_file():
        pytest.skip("Jetson CUDA toolkit unavailable")
    from argus.cuda_stereo import CudaStereoMatcher

    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, (96, 160), dtype=np.uint8)
    shift = 12
    right = np.zeros_like(left)
    right[:, :-shift] = left[:, shift:]
    disparity = CudaStereoMatcher().compute(
        left, right, max_disparity=32, radius=2, uniqueness_percent=5)
    valid = disparity[:, 40:-4]
    assert np.mean(valid == shift) > 0.9


def test_cuda_matcher_runs_on_non_creator_thread():
    pytest.importorskip("pycuda")
    if not __import__("pathlib").Path("/usr/local/cuda/bin/nvcc").is_file():
        pytest.skip("Jetson CUDA toolkit unavailable")
    from argus.cuda_stereo import CudaStereoMatcher

    matcher = CudaStereoMatcher()
    image = np.arange(96 * 160, dtype=np.uint8).reshape(96, 160)
    errors = []

    def run():
        try:
            matcher.compute(image, image, max_disparity=32, radius=2,
                            uniqueness_percent=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []

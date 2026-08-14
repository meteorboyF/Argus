"""Auditable CUDA stereo block matcher for the Jetson Orin (SM 8.7)."""
from __future__ import annotations

import numpy as np


CUDA_SOURCE = r'''
extern "C" __global__ void stereo_sad(
    const unsigned char *left, const unsigned char *right, float *output,
    int width, int height, int max_disparity, int radius, int uniqueness) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x < max_disparity + radius || x >= width - radius ||
        y < radius || y >= height - radius) {
        output[y * width + x] = -1.0f;
        return;
    }
    int best = 2147483647, second = 2147483647, best_d = -1;
    for (int d = 0; d < max_disparity; ++d) {
        int sad = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            const int row = (y + dy) * width;
            for (int dx = -radius; dx <= radius; ++dx) {
                sad += abs((int)left[row + x + dx] -
                           (int)right[row + x + dx - d]);
            }
        }
        if (sad < best) { second = best; best = sad; best_d = d; }
        else if (sad < second) { second = sad; }
    }
    output[y * width + x] =
        (best_d > 0 && best * 100 < second * (100 - uniqueness))
        ? (float)best_d : -1.0f;
}
'''


class CudaStereoMatcher:
    """Compile once, reuse buffers, and return disparity in working pixels."""

    def __init__(self, compute_arch: str = "87"):
        try:
            import pycuda.autoprimaryctx as cuda_context
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule
            self._cuda = cuda
            self._context = cuda_context.context
            module = SourceModule(
                CUDA_SOURCE, options=["-O3", f"-arch=sm_{compute_arch}"],
                nvcc="/usr/local/cuda/bin/nvcc")
            self._kernel = module.get_function("stereo_sad")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"CUDA stereo initialization failed: {exc}") from exc
        self._shape: tuple[int, int] | None = None
        self._left_dev = self._right_dev = self._output_dev = None
        self._output: np.ndarray | None = None

    def _allocate(self, shape: tuple[int, int]) -> None:
        if shape == self._shape:
            return
        self._output = np.empty(shape, dtype=np.float32)
        pixels = int(np.prod(shape))
        self._left_dev = self._cuda.mem_alloc(pixels)
        self._right_dev = self._cuda.mem_alloc(pixels)
        self._output_dev = self._cuda.mem_alloc(self._output.nbytes)
        self._shape = shape

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray,
                max_disparity: int, radius: int,
                uniqueness_percent: int) -> np.ndarray:
        if left_gray.shape != right_gray.shape or left_gray.ndim != 2:
            raise ValueError("CUDA stereo requires equal-size grayscale images")
        left = np.ascontiguousarray(left_gray, dtype=np.uint8)
        right = np.ascontiguousarray(right_gray, dtype=np.uint8)
        # The matcher is constructed on the main thread but used by the safety
        # thread. Make the retained primary context current on the caller for
        # every operation; a context current only on the creator thread causes
        # cuMemAlloc/launch to fail with "invalid device context".
        self._context.push()
        try:
            self._allocate(left.shape)
            height, width = left.shape
            if max_disparity <= 1 or max_disparity + radius >= width:
                raise ValueError("disparity search range does not fit the input width")
            self._cuda.memcpy_htod(self._left_dev, left)
            self._cuda.memcpy_htod(self._right_dev, right)
            self._kernel(
                self._left_dev, self._right_dev, self._output_dev,
                np.int32(width), np.int32(height), np.int32(max_disparity),
                np.int32(radius), np.int32(uniqueness_percent),
                block=(16, 16, 1), grid=((width + 15) // 16, (height + 15) // 16, 1))
            self._cuda.memcpy_dtoh(self._output, self._output_dev)
            return self._output.copy()
        finally:
            self._context.pop()

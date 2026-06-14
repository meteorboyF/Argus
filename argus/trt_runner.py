"""Minimal TensorRT engine runner (Jetson).

Loads a serialized .engine file and runs synchronous inference. Used for the
YOLO-World and (optionally) RAFT-Stereo engines built on-device with trtexec.

This module imports tensorrt and pycuda, which only exist on the Jetson
(JetPack). It is imported lazily by callers so the rest of the package works on a
PC without TensorRT installed.
"""
from __future__ import annotations

import numpy as np

try:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401  (initialises CUDA context)
    import pycuda.driver as cuda
    _TRT_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TRT_AVAILABLE = False


class TRTRunner:
    def __init__(self, engine_path: str):
        if not _TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT/pycuda not available. This runs on the Jetson only; "
                "install with the JetPack TensorRT and `pip install pycuda`."
            )
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = int(np.prod([d for d in shape if d > 0]))
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            io = {"name": name, "host": host, "dev": dev, "shape": shape, "dtype": dtype}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(io)
            else:
                self.outputs.append(io)

    def infer(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for inp in self.inputs:
            data = feeds.get(inp["name"])
            if data is None and len(self.inputs) == 1:
                data = next(iter(feeds.values()))
            np.copyto(inp["host"], np.ascontiguousarray(data, dtype=inp["dtype"]).ravel())
            cuda.memcpy_htod_async(inp["dev"], inp["host"], self.stream)
            self.context.set_tensor_address(inp["name"], int(inp["dev"]))

        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["dev"]))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        results = {}
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["dev"], self.stream)
        self.stream.synchronize()
        for out in self.outputs:
            results[out["name"]] = out["host"].reshape(
                [d for d in out["shape"] if d > 0]
            )
        return results

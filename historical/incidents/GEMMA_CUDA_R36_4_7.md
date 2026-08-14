# Research Request: Gemma 4 E2B Vision GPU Offload Fails on Jetson Orin Nano 8 GB

> **Resolved on 2026-08-14.** NVIDIA Jetson Linux was upgraded from R36.4.7 to
> R36.5.2, `llama.cpp` was rebuilt cleanly for CUDA architecture 87, and the
> previously failing full decoder offload (`--device CUDA0 -ngl 99`) loaded
> successfully. A live three-camera preview passed one wide-camera frame through
> the mandatory privacy gate (one face blurred); ARGUS correctly answered,
> "There is a person in front of you." Gemma took 6.90 s total: 5.77 s prompt
> evaluation plus 1.13 s for 10 output tokens (8.89 tokens/s). The historical
> R36.4.7 failure details below are retained as engineering evidence.

## Goal

Find a safe, reproducible way to run **Gemma 4 E2B multimodal inference with
meaningful GPU acceleration** on an NVIDIA Jetson Orin Nano Super 8 GB.

The model works correctly through native `llama.cpp` on CPU, including its
vision projector, but it is too slow for an assistive smart-glasses product.
CUDA offload fails during model loading even at very low layer counts.

We need researched solutions that apply specifically to **Jetson integrated
memory, L4T R36.4.7, CUDA 12.6, and current llama.cpp**. Desktop NVIDIA GPU
instructions are not automatically applicable.

## Product context

ARGUS is smart glasses for blind and visually impaired users:

- Fast safety loop: rule-based OpenCV SGBM stereo depth, approximately 10 Hz.
- Slow loop: wake word, Whisper STT, mandatory InsightFace privacy gate,
  Gemma 4 E2B vision agent, YOLO-World grounding, and Piper TTS.
- The fast safety loop must remain non-neural and independent of the VLM.
- The VLM must never receive an unblurred frame.

Repository: <https://github.com/meteorboyF/Argus>

## Exact device and software

- Device: NVIDIA Jetson Orin Nano Super, 8 GB unified memory
- GPU: Orin, compute capability 8.7
- CPU: 6-core ARM64
- OS: Ubuntu 22.04 ARM64
- JetPack: 6.2
- L4T at failure: R36.4.7; fixed/verified release: R36.5.2
- CUDA: 12.6
- TensorRT: 10.3.0
- Python: 3.10.12
- Native `llama.cpp`: recent source build, CUDA enabled for architecture 87
- Build configuration:

```bash
PATH="/usr/local/cuda/bin:$PATH" cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --config Release -j2
```

The native ARM64 build completes successfully. CUDA is detected correctly.
Jetson-specific PyTorch also reports `torch.cuda.is_available() == True`, and
the YOLO-World TensorRT engine works, so the GPU is not generally unavailable.

## Model artifacts

- Main model:
  `/opt/argus/models/gemma-4-E2B-it-Q4_K_M.gguf`
- Main model size: approximately 2.9 GB
- Vision projector:
  `/opt/argus/models/mmproj-gemma4-e2b-f16.gguf`
- Projector size: approximately 940 MB
- Runtime must use native `llama-server` with the projector, not a Python VLM
  wrapper and not a text-only substitute.

## Working CPU profile

This profile loads successfully and answers image questions correctly:

```bash
llama-server \
  --model /opt/argus/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --mmproj /opt/argus/models/mmproj-gemma4-e2b-f16.gguf \
  --device none \
  -ngl 0 \
  --parallel 1 \
  --fit off \
  --flash-attn off \
  --ctx-size 2048 \
  --jinja \
  --reasoning off \
  --no-mmproj-offload \
  --host 127.0.0.1 \
  --port 8080
```

Verified:

- `GET /health` returns `{"status":"ok"}`.
- Gemma receives a privacy-gated camera frame and gives coherent descriptions.
- The mmproj loads and processes single images correctly.

## Performance problem

Initial configuration used hidden reasoning, 1024-pixel images, and a 256-token
budget. Requests exceeded two consecutive 60-second timeouts and were estimated
at more than three minutes.

After tuning:

- Reasoning disabled.
- Input reduced to 256 maximum image side.
- Completion limited to 48 tokens.
- Automatic retry disabled.

Measurements:

- Warm-cache lower bound: 6.41 seconds for one short description.
- Honest cold production preview: 25.6 seconds.
- Cold server timing was approximately:
  - 12.7 seconds for prompt and image processing.
  - 12.8 seconds to generate 14 tokens.
  - Generation rate approximately 1.09 tokens/second in that request.

This remains too slow for an interactive assistive device.

## Memory pressure

With the CPU-resident Gemma server idle, a prior measurement showed:

```text
RAM 5528/7620 MB
SWAP 471/3810 MB
```

During later desktop use, total RAM reached approximately 6.1 GB with 2.7 GB
swap used. GNOME, VS Code, and previously Chrome contribute additional pressure,
but CUDA offload has also failed when several gigabytes were reported free at
process startup.

The final deployed device can be headless, but current development happens with
the graphical desktop and VS Code running.

## GPU attempts and exact failures

### Full offload (`-ngl 99`)

The server reported roughly 3.95 GB CUDA memory free at startup. Loading failed
when it requested an additional approximately 1.38 GB buffer:

```text
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1407.73 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 1476114816
llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

### Partial offload (`-ngl 20`, `--fit off`)

Also failed with CUDA/NVMAP out-of-memory while loading.

### Partial offload (`-ngl 4`, `--fit off`)

Failed despite the process reporting approximately 3.875 GB free at startup:

```text
CUDA0 : Orin (7619 MiB, 3875 MiB free)
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 345.72 MiB on device 0: cudaMalloc failed: out of memory
alloc_tensor_range: failed to allocate CUDA0 buffer of size 362517888
llama_model_load: error loading model: unable to allocate CUDA0 buffer
```

### Automatic fitting

Using partial offload with automatic fitting triggered a llama.cpp scheduler
assertion instead of finding a safe configuration:

```text
GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
```

Therefore the working CPU profile uses `--fit off`.

### CPU weights but default CUDA device

Using `-ngl 0` without explicitly setting `--device none` still caused CUDA
compute-buffer allocation. It failed on a roughly 121 MB CUDA allocation:

```text
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 120.63 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 126487552
graph_reserve: failed to allocate compute buffers
```

Adding `--device none` made the model load successfully.

## Additional llama.cpp multimodal issue observed

One request containing three separate images crashed the Gemma 4 projector:

```text
clip_image_batch_encode: output buffer has 153600 elements but expected 460800
Output buffer size mismatch
```

Single-image requests work. Combining temporal frames into one contact-sheet
image also works. This is separate from the GPU allocation problem but may be
relevant when selecting a llama.cpp version or patch.

## Relevant external reports already found

1. A llama.cpp user running Gemma 3n E2B on an Orin Nano reported the same
   `NvMapMemAllocInternalTagged error 12` and `cudaMalloc failed` behavior even
   when CUDA initially reported about 6.7 GB free:
   <https://github.com/ggml-org/llama.cpp/discussions/16706>

2. NVIDIA forum users reported `unable to allocate CUDA0 buffer` after updating
   Jetson Linux from R36.4.4 to R36.4.7. CPU mode continued to work:
   <https://forums.developer.nvidia.com/t/unable-to-allocate-cuda0-buffer-after-updating-ubuntu-packages/347862>

3. A separate llama.cpp issue reported incorrect or blank Gemma 3n CUDA output
   on Jetson Orin Nano even when model loading succeeded:
   <https://github.com/ggml-org/llama.cpp/issues/15034>

4. The official llama.cpp MobileVLM documentation reports much faster full-GPU
   multimodal inference on Orin, but that test uses MobileVLM 1.7B rather than
   Gemma 4 E2B:
   <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/MobileVLM.md>

These reports suggest, but do not prove, that L4T R36.4.7 may have a CUDA/NVMAP
allocation or fragmentation problem relevant to this device.

## Superseded low-risk experiment

Research suggests testing mmap-disabled loading after clearing filesystem
caches:

```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

Then launch with:

```bash
--no-mmap --no-mmproj-offload --device CUDA0 -ngl 1 --fit off --parallel 1
```

If one layer succeeds, increase `-ngl` incrementally while measuring RAM, swap,
CUDA allocations, correctness, and cold image-query latency. The CPU profile
must be restored immediately after any failure.

This experiment was not needed after the documented BSP fix. R36.5.2 succeeded
with full decoder offload, so `--no-mmap -ngl 1` should not be presented as the
solution to this incident.

## Constraints and non-solutions

- Do not install generic PyPI Torch; Jetson requires NVIDIA's CUDA wheel.
- Do not cross-compile TensorRT engines.
- Do not remove the mandatory face-blur privacy gate.
- Do not put a neural network in the fast safety loop.
- Do not claim that large swap makes CUDA allocations safe; Jetson CUDA/NVMAP
  behavior must be verified empirically.
- Do not assume `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` solves Jetson integrated
  memory allocation. External reports say it did not fix this failure.
- Reflashing or downgrading JetPack/L4T is disruptive and should be recommended
  only with strong evidence, exact compatible versions, rollback precautions,
  and an explanation of firmware/BSP compatibility.
- Replacing Gemma is acceptable only as a researched fallback comparison. The
  preferred solution is to accelerate the current Gemma 4 E2B stack safely.

## Questions for the researcher

Please investigate and answer with primary sources, exact versions, commands,
and known risks:

1. Is the CUDA/NVMAP allocation failure a confirmed regression or known issue
   in L4T R36.4.7 on Orin Nano 8 GB?
2. Is it fixed in a newer JetPack/L4T release, kernel, firmware, CUDA package,
   or llama.cpp commit?
3. Is R36.4.4, R36.4.3, JetPack 6.2.1, or another version known to work, and
   what is the safe upgrade/downgrade path?
4. Does `--no-mmap` materially change CUDA allocation success on Jetson, and
   why?
5. Are there required Jetson-specific llama.cpp build flags such as MMQ,
   cuBLAS, unified-memory, VMM, or graph settings?
6. Can the main model remain mmap/CPU-resident while only selected compute or
   decoder layers use CUDA without duplicating weights in unified memory?
7. Can the vision projector be quantized (for example Q8 or Q4) or replaced by
   an official smaller projector compatible with this exact Gemma 4 E2B GGUF?
8. Are `--image-min-tokens` and `--image-max-tokens` supported and reliable for
   Gemma 4 E2B, and what values preserve useful scene understanding?
9. Is there a known fix for the Gemma 4 multi-image output-buffer mismatch?
10. What is the fastest verified multimodal configuration on an **8 GB Orin
    Nano**, including model, quantization, runtime, RAM usage, and cold latency?
11. If Gemma 4 E2B cannot be made practical, which smaller VLM offers the best
    verified Jetson CUDA performance while supporting concise scene description
    and object awareness?

## Desired answer format

Please separate:

1. Confirmed facts and primary-source evidence.
2. Likely diagnoses.
3. Low-risk experiments in priority order.
4. Destructive/high-risk options such as reflashing.
5. Recommended production configuration and expected performance.

Do not recommend commands that alter firmware, boot configuration, kernel,
partitions, or the installed JetPack stack without clearly labeling them as
high risk and providing recovery steps.

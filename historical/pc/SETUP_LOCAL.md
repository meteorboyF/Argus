# ARGUS — Local PC Setup (Option C: PC Test Phase)

This guide sets up your **Windows PC** from scratch to run [`argus_pc_test.ipynb`](argus_pc_test.ipynb) on a **local VS Code Jupyter kernel** using the **GTX 1060 6GB**. No cloud, no Colab.

> **Hardware assumed:** Windows + Anaconda, NVIDIA GTX 1060 6GB (Pascal), ~160 GB free on D:.
> **Goal:** validate cameras, YOLO-World detection, and the speech stack locally — *before* any Jetson work.
> **NOT on this PC:** Gemma 4 E2B / any VLM (the "slow path" reasoning brain is deferred to the Jetson, per the pipeline doc §3 & §6.4).

---

## 0. The two-speed architecture (context)

- **Fast path (always on):** stereo depth (AR0234 pair) + **YOLO-World** open-vocabulary detection on the wide IMX477P camera. Continuous, non-ML safety reflex + sub-second grounding.
- **Slow path (on demand, Jetson only):** **Gemma 4 E2B** multimodal VLM orchestrates a `find_object(name)` tool call to YOLO-World, fuses the box with the depth map, and speaks guidance. **Never run on this PC.**

This PC test validates the fast-path perception + the speech I/O. Everything that runs on the 1060 here uses **FP32** — Pascal (compute capability 6.1) has no Tensor Cores, so FP16/AMP gives no speedup.

---

## 1. Conda environment + PyTorch (CUDA build for the 1060)

The GTX 1060 is **Pascal (compute capability 6.1)**. Modern PyTorch CUDA 12.x wheels still support it.

```powershell
# Recommended: a dedicated env (or use base if you prefer)
conda create -n argus python=3.11 -y
conda activate argus
```

> Python **3.11** is the safest choice — some audio/onnx wheels lag on 3.12/3.13.

**Install the CUDA build of PyTorch (NOT the CPU wheel):**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify CUDA is visible:**

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected: `2.x.x True NVIDIA GeForce GTX 1060 6GB`. If `False`, see Troubleshooting → "CUDA not found".

**Install the rest:**

```powershell
pip install ultralytics opencv-python faster-whisper openwakeword onnx onnxruntime sounddevice soundfile piper-tts
```

Notes:
- `ultralytics` provides YOLO-World (`YOLOWorld`) and downloads `yolov8s-worldv2.pt` on first use.
- `faster-whisper` runs Whisper on CPU with INT8 (CTranslate2) — light and fast.
- `onnxruntime` (CPU) only *validates* the ONNX export; the 1060 runs YOLO via torch.

---

## 2. Arducam driver / UVC setup (Windows, USB 3.0)

Both Arducam models are **USB UVC** cameras — Windows uses the built-in UVC driver; no special install needed in most cases.

- **2× Arducam AR0234** (2.3 MP global shutter) — your **stereo pair** (geometry/distance).
- **1× Arducam IMX477P** (12 MP, wide M12 lens) — your **wide scene camera** (YOLO-World + VLM snapshots).

Steps:
1. Plug each camera into a **USB 3.0** port (blue). Prefer **separate USB controllers/buses** — three cameras on one hub can saturate bandwidth.
2. Open the Windows **Camera** app or Device Manager → "Cameras" to confirm all three enumerate.
3. If a camera enumerates but won't stream in OpenCV, install **Arducam's UVC tooling** from arducam.com (only needed for advanced controls/firmware). Standard streaming works with the stock UVC driver.
4. The AR0234 global-shutter pair supports **hardware sync** (external trigger) for sub-ms alignment — that's a Jetson-phase concern. On the PC we measure **software sync** delta (notebook Cell 4); under ~15 ms is acceptable for validation.

**Bandwidth tip:** the IMX477P at full 12 MP is heavy. You don't need full resolution for detection — ~1920×1080 is plenty and lighter on the bus.

---

## 3. VS Code: select the LOCAL kernel (do NOT use Colab)

1. Open the repo folder in VS Code.
2. Install the **Python** and **Jupyter** extensions (Microsoft) if not present.
3. Open `argus_pc_test.ipynb`.
4. Top-right **"Select Kernel"** → **Python Environments** → pick your conda env (`argus`, or `base`).
5. Confirm the kernel indicator shows your conda env, not a remote/Colab one.

> ⚠️ **DO NOT** use the "Colab" extension or any cloud/remote kernel. This project runs **locally** so the USB cameras and the 1060 are accessible. **A cloud kernel cannot see your USB cameras at all.**

---

## 4. Running `argus_pc_test.ipynb` — cell by cell

Run **top to bottom**. What success looks like:

| Cell | What it does | Success |
|---|---|---|
| 1 | Env detect + folders + GPU check | `Environment: LOCAL`, `CUDA avail: True`, `GPU: ...GTX 1060...`, cap `6.1`; folders under `D:\ARGUS` |
| 2 | Dependency check/install | every package prints a version, no FAIL |
| 3 | Camera enumeration | **3** indices OPEN; note which resolutions = which camera |
| 4 | 3-cam live + stereo sync | 3 sample JPGs in `D:\ARGUS\logs`; stereo delta `< ~15 ms` |
| 5 | YOLO-World detection | your named objects detected with boxes; `yoloworld_detection.jpg` saved |
| 6 | Fine-tune (gated) | prints "skipping" (keep `RUN_FINETUNE = False`) |
| 7 | ONNX export | `yoloworld_640.onnx` in `exports\onnx`; onnxruntime prints inputs/outputs |
| 8 | Speech stack | wake model loads; your speech → text; Piper speaks |
| 9 | Manifest | `pc_test_manifest.json` written in `models\` |

**Before Cell 4/5:** edit the indices (`LEFT_IDX`, `RIGHT_IDX`, `WIDE_IDX`) to match Cell 3's output.

**Before Cell 8 Piper:** download a Piper voice (`en_US-lessac-medium.onnx` + `.json`) from
`https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium`
and place both files in `D:\ARGUS\models\piper\`.

---

## 5. Troubleshooting

### Camera not detected (Cell 3 shows fewer than 3)
- Try a different **USB 3.0** port; move cameras to **separate buses** (bandwidth).
- Close any app using the camera (Windows Camera, Zoom, Teams).
- The notebook uses `cv2.CAP_DSHOW`. If still failing, try `cv2.CAP_MSMF` by editing the backend in the cell.
- Device Manager → Cameras: all three should appear with no warning icons.
- A **powered** USB 3.0 hub helps if your ports can't supply enough current for three cameras.

### CUDA not found on the 1060 (Cell 1 prints `CUDA avail: False`)
- You likely installed the **CPU** torch wheel. Reinstall the CUDA build:
  ```powershell
  pip uninstall -y torch torchvision
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- Update your **NVIDIA driver** (a recent Game Ready / Studio driver covers Pascal + the CUDA 12 runtime).
- You do **not** need a separate CUDA Toolkit install — the pip wheel bundles the runtime.
- Confirm the GPU is active: `nvidia-smi` should list the GTX 1060.

### ONNX export errors (Cell 7)
- Ensure `dynamo=False` (already set). The dynamo exporter is unreliable on vision ops (Phase-0 rule).
- If `simplify=True` fails, set it to `False` and re-run — simplification is optional.
- Keep `opset=17`. Drifting opsets can break the later TensorRT import on the Jetson.

### Out of memory on the 1060 (Cell 5/6)
- 6 GB is tight. For detection, use one camera at a time. For fine-tune (Cell 6), keep `batch=2`, `imgsz=640`, `amp=False`.
- Close other GPU apps (browsers with hardware accel, games).

### Speech: no audio / no mic (Cell 8)
- Set `RECORD_MIC = False` to skip live capture and test model loading + Piper TTS only.
- Windows Sound settings → Input/Output devices enabled and not exclusive-locked.

---

## 6. After the PC test passes

1. `pc_test_manifest.json` records every artifact + its Jetson destination.
2. Next phase = **Jetson Orin Nano Super**: JetPack 6, build TensorRT engines on-device (`trtexec --fp16`), launch Gemma 4 E2B with vision via native llama.cpp on the slow path.
3. The ONNX you exported here transfers to the Jetson; **TensorRT engines must be built on the Jetson** (device-specific).

See [`README.md`](README.md) and [`docs/ARGUS_Final_Pipeline.pdf`](docs/ARGUS_Final_Pipeline.pdf) for the full architecture.

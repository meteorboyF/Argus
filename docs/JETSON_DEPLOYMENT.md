# ARGUS — Jetson Orin Nano Super Deployment

End-to-end bring-up of the ARGUS runtime on the device. Follow top to bottom.
Every step says how to **verify** it before moving on. If you are driving this
with Claude Code, start from [JETSON_PROMPT_01.md](JETSON_PROMPT_01.md).

> The Jetson runs **Ubuntu 22.04 / ARM64 (JetPack 6)**. All commands are Linux.
> There is **no Colab and no Windows** here — the `.bat` file is for the PC only.
>
> Open work items that are NOT finished in this repo are tracked in
> [KNOWN_GAPS.md](KNOWN_GAPS.md) — read it before assuming something works.

---

## 0. What you need before starting

| Item | Notes |
|---|---|
| Jetson Orin Nano Super 8 GB | flashed with **JetPack 6** (CUDA, cuDNN, TensorRT included) |
| 2× Arducam AR0234 USB | stereo pair, any mounting position/angle — calibration adapts |
| 1× Arducam IMX477P USB | wide scene camera |
| USB microphone + speaker/bone-conduction | audio I/O (`arecord -l` / `aplay -l` to confirm) |
| Flat printed checkerboard | any common size (9×6 inner corners typical); glued to stiff card |
| Internet access | first install downloads models (~4–5 GB total) |
| ≥ 15 GB free disk | models + llama.cpp build + pip caches |

---

## 1. First-boot device prep

```bash
# Max performance (re-run after every reboot, or add to a boot service):
sudo nvpmodel -m 0
sudo jetson_clocks

# Swap: 8 GB unified memory needs headroom for builds and model load spikes.
swapon --show                     # if empty, enable zram (JetPack default):
sudo systemctl enable --now nvzramconfig
```

---

## 2. Clone + install

```bash
cd ~
git clone https://github.com/meteorboyF/Argus.git
cd Argus
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

The installer is idempotent. It creates `/opt/argus/{models,engines,exports,config,logs}`,
installs apt + Python deps, builds **llama.cpp with CUDA (arch 87, -j4)**, installs
the `argus` package editable, and seeds `/opt/argus/config/argus.yaml`.

### 2a. torch on Jetson (critical — do not skip verification)

`pip install torch` gives a CPU-only ARM build. You need NVIDIA's Jetson wheel:

```bash
# Option A — Jetson AI Lab index (pick the cuXXX matching your JetPack 6.x):
python3 -m pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# Option B — NVIDIA's redist page (download the .whl matching your JetPack):
#   https://developer.download.nvidia.com/compute/redist/jp/
```

**Verify (must print True):**
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If the index URL 404s, the JetPack minor version moved — check
`https://pypi.jetson-ai-lab.io` for the current path, or use Option B.

### 2b. onnxruntime (privacy gate)

`requirements-jetson.txt` installs **CPU** `onnxruntime` (there is **no**
`onnxruntime-gpu` aarch64 wheel on PyPI — do not try, pip will fail). CPU is
fast enough for SCRFD face detection on the slow path. Optional GPU upgrade:
install NVIDIA's Jetson onnxruntime-gpu wheel from the Jetson AI Lab index or
[elinux.org Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime), then the
privacy gate picks up `CUDAExecutionProvider` automatically.

---

## 3. Model artefacts

```bash
./scripts/download_models.sh
```

Automatic: YOLO-World weights, the Piper voice, openWakeWord models.
**Manual (the script prints instructions):** the Gemma vision GGUF + its
`mmproj` projector. Pick an INT4 (`Q4_K_M`) multimodal Gemma GGUF on Hugging
Face, download both files into `/opt/argus/models/`, and either rename them to
the config defaults or point these at the real filenames:

- `/opt/argus/config/argus.yaml` → `agent.model_gguf`, `agent.mmproj_gguf`
- `scripts/run_llama_server.sh` → `MODEL`, `MMPROJ`

Expected inventory after this step:

```
/opt/argus/models/yolov8s-worldv2.pt
/opt/argus/models/gemma-4-E2B-it-Q4_K_M.gguf        (or your actual filename)
/opt/argus/models/mmproj-gemma4-e2b-f16.gguf         (or your actual filename)
/opt/argus/models/piper/en_US-lessac-medium.onnx (+ .json)
```

First runs of insightface (privacy) and faster-whisper (STT) download their own
small models automatically — make sure the first `selftest`/`run` has internet.

---

## 4. TensorRT engines (optional at first — do not block on this)

The runtime works without engines: depth defaults to SGBM (CPU) and YOLO-World
runs its `.pt` via torch/CUDA. When you have ONNX exports in `/opt/argus/exports/`:

```bash
./scripts/build_engines.sh     # trtexec --fp16, engines land in /opt/argus/engines
```

Engines are **device-specific** — always built here, never copied from a PC.
Note: the YOLO-World TRT engine is not yet wired into the grounding code path
(see [KNOWN_GAPS.md](KNOWN_GAPS.md)); the `.pt` path is the working default.

---

## 5. Cameras: discovery + smart calibration

### 5a. Confirm enumeration

```bash
v4l2-ctl --list-devices
```

You do **not** need to record indices. The runtime discovers cameras by V4L2
device name (`AR0234` / `IMX477` hints in `argus.yaml`) with a resolution-grouping
fallback, and re-binds left/right to physical USB ports after calibration. If
the hints don't match what `v4l2-ctl` shows for your units, edit
`camera.stereo_name_hint` / `camera.wide_name_hint`.

### 5b. Calibrate (adapts to ANY mounting position)

The stereo cameras can be mounted at any angle/baseline — toed-out, non-coplanar,
whatever the frame dictates. Calibration measures the actual geometry and depth
is rectified to match. **Left/right order doesn't matter either** — a swapped
pair is detected from the recovered geometry and corrected automatically.

```bash
# with a display attached:
python3 scripts/calibrate_stereo.py --square-mm 25
# over SSH (no display):
python3 scripts/calibrate_stereo.py --square-mm 25 --headless
```

Move the checkerboard slowly around the whole field of view (corners, centre,
near, far, tilted). Auto-capture fires when the board is steady and visible in
both cameras. Targets: **RMS < 0.6 px**, **vertical error < 1.0 px**.

The result (`/opt/argus/config/stereo_calib.npz`) contains rectification maps,
the Q reprojection matrix, baseline/focal, and the **USB port paths** of the
left/right cameras. No config edits needed — `argus/depth.py` and
`argus/cameras.py` load it automatically.

### 5c. Verify with a tape measure

```bash
python3 scripts/calibrate_stereo.py --verify            # add --headless over SSH
```

Aim at a wall 0.5–3 m away; the printed centre depth should match reality
within ~5–10 %. If it doesn't, re-run calibration with better coverage/lighting.

Full details: [CALIBRATION.md](CALIBRATION.md). **Re-calibrate whenever a camera
is re-seated or the frame flexes.**

---

## 6. Audio

```bash
arecord -l        # mic present?
aplay -l          # output present?
python3 -m sounddevice    # names/indices as the runtime sees them
```

If the defaults are wrong, set `speech.input_device` / `speech.output_device`
in `/opt/argus/config/argus.yaml` (index or name substring, e.g. `"USB"`).
Wake word is `hey_jarvis` (openWakeWord built-in) until a custom "ARGUS" model
is trained (see KNOWN_GAPS).

---

## 7. Start the Gemma server

```bash
./scripts/run_llama_server.sh        # binds 127.0.0.1:8080, --jinja enabled
```

Leave it running (tmux, or the systemd unit below). Verify:

```bash
curl http://127.0.0.1:8080/health
# and a real completion:
curl http://127.0.0.1:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'
```

Watch `tegrastats` during the first load — the GGUF + projector should settle
around 2.5–3 GB.

---

## 8. Self-test, then run

```bash
python3 -m argus selftest
```

Work through every FAIL. The self-test checks: deps, torch CUDA, TensorRT,
model files, **calibration quality + resolution match**, camera discovery,
audio devices, the privacy gate, and the llama server.

```bash
# Stage 1 — fast safety loop only (no mic/speaker). Walk toward a wall:
python3 -m argus run --no-audio
# Stage 2 — one text-driven slow-path turn (needs the llama server):
python3 -m argus query "what is in front of me?"
# Stage 3 — the full system (wake word "hey jarvis"):
python3 -m argus run
```

> The privacy gate is enforced: with `privacy.require_gate: true` (default) the
> runtime refuses to start the agent path if face blurring failed to initialise.

---

## 9. Run at boot (optional)

`/etc/systemd/system/argus-llm.service`:
```ini
[Unit]
Description=ARGUS Gemma llama.cpp server
After=network.target

[Service]
Environment=ARGUS_HOME=/opt/argus
ExecStart=/opt/argus/llama.cpp/build/bin/llama-server --model /opt/argus/models/gemma-4-E2B-it-Q4_K_M.gguf --mmproj /opt/argus/models/mmproj-gemma4-e2b-f16.gguf -ngl 99 --flash-attn on --ctx-size 2048 --jinja --host 127.0.0.1 --port 8080
Restart=on-failure
User=YOUR_USER

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/argus.service`:
```ini
[Unit]
Description=ARGUS runtime
After=argus-llm.service
Requires=argus-llm.service

[Service]
Environment=ARGUS_HOME=/opt/argus
ExecStartPre=/bin/sleep 20
ExecStart=/usr/bin/python3 -m argus run
Restart=on-failure
User=YOUR_USER

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now argus-llm argus
```

---

## Memory budget (watch with `tegrastats`)

| Component | ~Memory |
|---|---|
| Gemma INT4 GGUF + mmproj | 2.5–3.0 GB |
| YOLO-World .pt (torch, loaded) | 0.6–1.0 GB |
| Privacy gate (SCRFD, CPU ORT) | 0.2 GB |
| Speech stack (CPU) | 0.3 GB |
| SGBM depth + buffers | 0.2 GB |
| OS + desktop | 1.0–1.5 GB (disable the GUI: `sudo systemctl set-default multi-user.target`) |

If you brush 8 GB: disable the desktop, confirm zram is on, and reduce
`agent.ctx_size`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` False | Install the NVIDIA Jetson torch wheel (§2a), not pip default |
| `pip install onnxruntime-gpu` fails | Expected on ARM — use CPU `onnxruntime` (§2b) |
| Cameras not found / wrong ones picked | `v4l2-ctl --list-devices`; fix `stereo_name_hint`/`wide_name_hint`, or set `auto_detect: false` + explicit indices |
| Left/right look mirrored / depth inverted | Re-run calibration — the swap check fixes assignment automatically |
| Depth wildly wrong | `calibrate_stereo.py --verify`; re-calibrate at the runtime resolution with better board coverage |
| Fast loop misses 10 Hz | Increase `depth.fast_downscale` (2 → 3), or lower `stereo_width/height` and re-calibrate |
| Constant false "step down" warnings | Raise `safety.floor_drop_invalid_fraction`; confirm calibration valid-pixel % in `--verify` |
| Wake word never triggers | `python3 -m sounddevice` → set `speech.input_device`; lower `wake_threshold`; speak "hey jarvis" clearly |
| Agent never calls find_object | Keep `tool_protocol: prompt` (default). `native` needs a tool-capable chat template |
| llama server unreachable | Start `run_llama_server.sh`; check `/health`; confirm the GGUF paths/filenames |
| `RuntimeError: Privacy gate failed to initialise` | Fix insightface/onnxruntime (selftest §1); first run needs internet to fetch `buffalo_s` |
| Out of memory | zram on, GUI off, `nvpmodel -m 0`, smaller ctx_size, check for duplicate llama-server |
| TRT engine load fails | Rebuild on this device; depth auto-falls back to SGBM |

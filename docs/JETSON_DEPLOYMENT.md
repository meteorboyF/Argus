# ARGUS — Jetson Orin Nano Super Deployment

End-to-end bring-up of the ARGUS runtime on the device. Follow top to bottom.

> The Jetson runs **Ubuntu 22.04 / ARM64 (JetPack 6)**. All commands are Linux.
> There is **no Colab and no Windows** here — the `.bat` file is for the PC only.

---

## 0. Prerequisites

- Jetson Orin Nano Super flashed with **JetPack 6** (includes CUDA, cuDNN, TensorRT).
- Set max performance once per boot session:
  ```bash
  sudo nvpmodel -m 0      # max power mode
  sudo jetson_clocks      # lock clocks high
  ```
- The three cameras connected over USB 3.0 (prefer separate buses).
- Network access for the first install.

---

## 1. Clone the repo

```bash
cd ~
git clone https://github.com/meteorboyF/Argus.git
cd Argus/ARGUS
```

---

## 2. Run the installer

```bash
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

This creates `/opt/argus/{models,engines,exports,config,logs}`, installs system +
Python deps, builds **llama.cpp with CUDA**, installs the `argus` package, and
seeds the config. It is idempotent.

### torch on Jetson (important)
`pip install torch` does **not** give you a CUDA build on ARM. Use NVIDIA's
Jetson wheel matching your JetPack from
`https://developer.download.nvidia.com/compute/redist/jp/` (or the version that
ships in your L4T image). Verify:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. Copy model artefacts

Place these under `/opt/argus` (from the PC export, Drive, or `scp`):

```
/opt/argus/models/yolov8s-worldv2.pt              # from the PC test
/opt/argus/exports/yoloworld_640.onnx             # from the PC test (dynamo=False)
/opt/argus/models/gemma-4-E2B-it-Q4_K_M.gguf      # Gemma 4 E2B quantised
/opt/argus/models/mmproj-gemma4-e2b-f16.gguf      # Gemma 4 vision projector
/opt/argus/models/piper/en_US-lessac-medium.onnx  # + the matching .json
```

Example transfer from the PC:
```bash
# on the PC
scp D:\ARGUS\exports\onnx\yoloworld_640.onnx  user@jetson:/opt/argus/exports/
scp D:\ARGUS\models\yolov8s-worldv2.pt        user@jetson:/opt/argus/models/
```

Gemma 4 E2B GGUF + mmproj: download the quantised checkpoint and vision projector
(e.g. from the Hugging Face GGUF repo) directly on the Jetson with `wget`/`huggingface-cli`.

---

## 4. Build TensorRT engines (on the device)

```bash
./scripts/build_engines.sh
```

Produces `/opt/argus/engines/yoloworld_640_fp16.engine` (and RAFT-Stereo if its
ONNX is present). Engines are **device-specific** — this must run on the Jetson.

---

## 5. Calibrate the stereo cameras

```bash
python scripts/calibrate_stereo.py --left 0 --right 1 \
    --rows 6 --cols 9 --square-mm 25 \
    --out /opt/argus/config/stereo_calib.npz
```

Then copy the printed `baseline_m` and `focal_px` into
`/opt/argus/config/argus.yaml` under `depth:`. Find camera indices with:
```bash
v4l2-ctl --list-devices
```

---

## 6. Start the Gemma 4 server

```bash
./scripts/run_llama_server.sh        # serves vision LLM on :8080
```
Leave it running (use `tmux`/`systemd` for a persistent service). Health check:
```bash
curl http://127.0.0.1:8080/health
```

---

## 7. Self-test, then run

```bash
python -m argus selftest             # green-lights deps, models, cameras, server
python -m argus run                  # full two-speed runtime
# or, fast loop only (no mic/speaker):
python -m argus run --no-audio
# or, single interaction for debugging:
python -m argus query "what is in front of me?"
```

---

## Run at boot (optional)

Create `/etc/systemd/system/argus.service`:
```ini
[Unit]
Description=ARGUS runtime
After=network.target

[Service]
Environment=ARGUS_HOME=/opt/argus
ExecStartPre=/bin/bash -c '/opt/argus/llama.cpp/build/bin/llama-server --model /opt/argus/models/gemma-4-E2B-it-Q4_K_M.gguf --mmproj /opt/argus/models/mmproj-gemma4-e2b-f16.gguf -ngl 99 --flash-attn on --ctx-size 2048 --no-mmproj-offload --port 8080 &'
ExecStart=/usr/bin/python3 -m argus run
Restart=on-failure
User=YOUR_USER

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now argus
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` False | Install NVIDIA Jetson torch wheel (not pip default) |
| Cameras not found | `v4l2-ctl --list-devices`; fix indices in `argus.yaml`; separate USB buses |
| `trtexec: not found` | `/usr/src/tensorrt/bin/trtexec` — ships with JetPack |
| llama server unreachable | Start `run_llama_server.sh`; check `/health`; confirm GGUF paths |
| Out of memory | Ensure INT4 Gemma; close other GPU procs; `nvpmodel -m 0` then retry |
| No audio | Check `arecord -l` / `aplay -l`; run `--no-audio` to isolate |
| TRT engine load fails | Rebuild on this device; depth auto-falls back to SGBM |

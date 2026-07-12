#!/usr/bin/env bash
# =============================================================================
# ARGUS — Jetson Orin Nano Super dependency installer (JetPack 6 / Ubuntu 22.04)
# =============================================================================
# Run this ONCE on the Jetson after cloning the repo. It is idempotent — safe to
# re-run. It installs system packages, Python deps, and builds llama.cpp with
# CUDA for Gemma vision inference.
#
#   cd ~/Argus
#   chmod +x scripts/*.sh
#   ./scripts/setup_jetson.sh
#
# NOTE: This is a Linux/ARM64 shell script — the Jetson runs Ubuntu, not Windows.
# A Windows .bat cannot run here. For PC-side setup use SETUP_LOCAL.md instead.
# =============================================================================
set -euo pipefail

ARGUS_HOME="${ARGUS_HOME:-/opt/argus}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================"
echo " ARGUS Jetson setup"
echo "   repo:       $REPO_DIR"
echo "   ARGUS_HOME: $ARGUS_HOME"
echo "================================================================"

# ----------------------------------------------------------------- 0. sanity
if ! command -v nvcc >/dev/null 2>&1 && [ ! -d /usr/local/cuda ]; then
  echo "WARNING: CUDA not found. Make sure JetPack 6 is flashed before continuing."
fi

# The Orin Nano has 8 GB unified memory. Compiling llama.cpp and, later, running
# all models needs swap headroom. Recommend zram/swap if none is active.
if [ "$(swapon --show | wc -l)" -eq 0 ]; then
  echo "WARNING: no swap active. Strongly recommended on 8 GB:"
  echo "  sudo systemctl enable --now nvzramconfig  (JetPack default zram)"
  echo "  or create a swapfile: sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile \\"
  echo "     && sudo mkswap /swapfile && sudo swapon /swapfile"
fi

# ----------------------------------------------------------------- 1. dirs
echo "[1/7] Creating $ARGUS_HOME tree (may need sudo)..."
sudo mkdir -p "$ARGUS_HOME"/{models,models/piper,engines,exports,config,logs}
sudo chown -R "$USER":"$USER" "$ARGUS_HOME"

# ----------------------------------------------------------------- 2. apt
echo "[2/7] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
  python3-pip python3-dev python3-venv \
  build-essential cmake git pkg-config \
  libopenblas-dev libportaudio2 portaudio19-dev \
  libgl1 libglib2.0-0 \
  v4l-utils ffmpeg \
  alsa-utils \
  curl wget unzip

# ----------------------------------------------------------------- 3. pip deps
echo "[3/7] Installing Python dependencies..."
python3 -m pip install --upgrade pip
# NOTE: torch/torchvision for Jetson come from NVIDIA's index, NOT pip's default.
# If torch is already provided by your JetPack/L4T image, this is skipped.
if ! python3 -c "import torch" 2>/dev/null; then
  echo "      torch not found — install the NVIDIA Jetson wheel matching your"
  echo "      JetPack (see docs/JETSON_DEPLOYMENT.md section 2). Continuing with the rest."
fi
python3 -m pip install -r "$REPO_DIR/requirements-jetson.txt"
# pycuda for the TensorRT runner
python3 -m pip install pycuda || echo "      pycuda install failed — TRT runner unavailable until fixed"

# ----------------------------------------------------------------- 4. llama.cpp
echo "[4/7] Building llama.cpp with CUDA (for Gemma vision)..."
LLAMA_DIR="$ARGUS_HOME/llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
pushd "$LLAMA_DIR" >/dev/null
  git pull --ff-only || true
  # CUDA arch 87 = Orin. -j4 (not nproc) — a -j6 CUDA build can OOM 8 GB.
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j 4
popd >/dev/null
echo "      llama.cpp built at $LLAMA_DIR/build/bin"

# ----------------------------------------------------------------- 5. config
echo "[5/7] Seeding config..."
if [ ! -f "$ARGUS_HOME/config/argus.yaml" ]; then
  cp "$REPO_DIR/config/argus.yaml" "$ARGUS_HOME/config/argus.yaml"
  echo "      copied default config -> $ARGUS_HOME/config/argus.yaml"
fi

# ----------------------------------------------------------------- 6. argus pkg
echo "[6/7] Installing the argus package (editable)..."
python3 -m pip install -e "$REPO_DIR"

# ----------------------------------------------------------------- 7. summary
echo "[7/7] Done."
echo "================================================================"
echo " Next steps (details: docs/JETSON_DEPLOYMENT.md):"
echo "  1. Verify torch CUDA:  python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "  2. Fetch models:       ./scripts/download_models.sh   (+ Gemma GGUF manually)"
echo "  3. Build TRT engines:  ./scripts/build_engines.sh     (optional at first)"
echo "  4. Start the LLM:      ./scripts/run_llama_server.sh"
echo "  5. Calibrate cameras:  python3 scripts/calibrate_stereo.py --square-mm 25"
echo "  6. Self-test:          python3 -m argus selftest"
echo "  7. Run:                python3 -m argus run"
echo "================================================================"

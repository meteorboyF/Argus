#!/usr/bin/env bash
# =============================================================================
# ARGUS — Jetson Orin Nano Super dependency installer (JetPack 6 / Ubuntu 22.04)
# =============================================================================
# Run this ONCE on the Jetson after cloning the repo. It is idempotent — safe to
# re-run. It installs system packages, Python deps, and builds llama.cpp with
# CUDA for Gemma 4 vision inference.
#
#   cd ~/Argus/ARGUS
#   chmod +x scripts/setup_jetson.sh
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
  curl wget unzip

# ----------------------------------------------------------------- 3. pip deps
echo "[3/7] Installing Python dependencies..."
python3 -m pip install --upgrade pip
# NOTE: torch/torchvision for Jetson come from NVIDIA's index, NOT pip's default.
# If torch is already provided by your JetPack/L4T image, this is skipped.
if ! python3 -c "import torch" 2>/dev/null; then
  echo "      torch not found — install the NVIDIA Jetson wheel matching your"
  echo "      JetPack from https://developer.download.nvidia.com/compute/redist/jp/"
  echo "      (see docs/JETSON_DEPLOYMENT.md). Continuing with the rest."
fi
python3 -m pip install -r "$REPO_DIR/requirements-jetson.txt"
# pycuda for the TensorRT runner
python3 -m pip install pycuda || echo "      pycuda install failed — TRT runner unavailable until fixed"

# ----------------------------------------------------------------- 4. llama.cpp
echo "[4/7] Building llama.cpp with CUDA (for Gemma 4 vision)..."
LLAMA_DIR="$ARGUS_HOME/llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
pushd "$LLAMA_DIR" >/dev/null
  git pull --ff-only || true
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j "$(nproc)"
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
echo " Next steps:"
echo "  1. Copy model artefacts into $ARGUS_HOME/models and exports:"
echo "       - yolov8s-worldv2.pt, yoloworld_640.onnx"
echo "       - gemma-4-E2B-it-Q4_K_M.gguf, mmproj-gemma4-e2b-f16.gguf"
echo "       - piper/en_US-lessac-medium.onnx(.json)"
echo "  2. Build TensorRT engines:   ./scripts/build_engines.sh"
echo "  3. Start the LLM server:     ./scripts/run_llama_server.sh"
echo "  4. Calibrate the cameras:    python scripts/calibrate_stereo.py"
echo "  5. Self-test:                python -m argus selftest"
echo "  6. Run:                      python -m argus run"
echo "================================================================"

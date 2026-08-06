#!/usr/bin/env bash
# Fetch the model artefacts ARGUS needs into $ARGUS_HOME (default /opt/argus).
# Everything except the Gemma GGUF is fully automatic. Idempotent.
#
#   ./scripts/download_models.sh
set -euo pipefail

ARGUS_HOME="${ARGUS_HOME:-/opt/argus}"
MODELS="$ARGUS_HOME/models"
mkdir -p "$MODELS/piper"

# ---- YOLO-World weights (ultralytics official release asset) ----
YOLO_PT="$MODELS/yolov8s-worldv2.pt"
if [ ! -f "$YOLO_PT" ]; then
  echo "Downloading YOLO-World (yolov8s-worldv2.pt)..."
  wget -q --show-progress -O "$YOLO_PT" \
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt"
else
  echo "YOLO-World weights already present."
fi

# ---- Piper voice (en_US-lessac-medium) ----
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
for f in en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json; do
  if [ ! -f "$MODELS/piper/$f" ]; then
    echo "Downloading Piper voice: $f ..."
    wget -q --show-progress -O "$MODELS/piper/$f" "$PIPER_BASE/$f"
  fi
done

# ---- openWakeWord pretrained models (cached by the library) ----
python3 - <<'EOF' || echo "openWakeWord pre-download skipped (will download on first run)"
import openwakeword
openwakeword.utils.download_models()
print("openWakeWord models cached.")
EOF

# ---- Gemma GGUF + vision projector (manual: pick the exact repo/quant) ----
GGUF="$MODELS/gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ="$MODELS/mmproj-gemma4-e2b-f16.gguf"
if [ -f "$GGUF" ] && [ -f "$MMPROJ" ]; then
  echo "Gemma GGUF + mmproj already present."
else
  cat <<EOF

--------------------------------------------------------------------
MANUAL STEP — Gemma reasoning model (vision GGUF, ~2-3 GB):
Download an INT4 (Q4_K_M) multimodal Gemma GGUF *and its mmproj vision
projector* from Hugging Face into $MODELS, e.g. with:

  pip install -U "huggingface_hub[cli]"
  huggingface-cli download <REPO> <MODEL>.gguf  --local-dir $MODELS
  huggingface-cli download <REPO> mmproj-*.gguf --local-dir $MODELS

Then either rename the files to match the config defaults:
  $GGUF
  $MMPROJ
or point agent.model_gguf / agent.mmproj_gguf in
$ARGUS_HOME/config/argus.yaml (and scripts/run_llama_server.sh) at the
actual filenames. Verify it loads with scripts/run_llama_server.sh.
--------------------------------------------------------------------
EOF
fi

echo
echo "Model inventory under $MODELS:"
find "$MODELS" -maxdepth 2 -type f -printf "  %-60p %10s bytes\n" 2>/dev/null || ls -lR "$MODELS"

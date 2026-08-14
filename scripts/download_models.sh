#!/usr/bin/env bash
# Fetch the model artefacts ARGUS needs into $ARGUS_HOME (default /opt/argus).
# Everything except the Gemma GGUF is fully automatic. Idempotent.
#
#   ./scripts/download_models.sh
set -euo pipefail

# SHA-256 identities of the audited device artifacts. Downloads or manually
# supplied files must match these exact bytes unless STATUS.md is deliberately
# updated in the same verified feature commit.
YOLO_PT_SHA256="9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"
PIPER_ONNX_SHA256="5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f"
PIPER_JSON_SHA256="efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0"
GEMMA_SHA256="9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d"
MMPROJ_SHA256="140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa"

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
echo "$YOLO_PT_SHA256  $YOLO_PT" | sha256sum --check --status || {
  echo "YOLO-World checksum mismatch: $YOLO_PT" >&2; exit 1;
}

# ---- Piper voice (en_US-lessac-medium) ----
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
for f in en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json; do
  if [ ! -f "$MODELS/piper/$f" ]; then
    echo "Downloading Piper voice: $f ..."
    wget -q --show-progress -O "$MODELS/piper/$f" "$PIPER_BASE/$f"
  fi
done
echo "$PIPER_ONNX_SHA256  $MODELS/piper/en_US-lessac-medium.onnx" | sha256sum --check --status || {
  echo "Piper ONNX checksum mismatch" >&2; exit 1;
}
echo "$PIPER_JSON_SHA256  $MODELS/piper/en_US-lessac-medium.onnx.json" | sha256sum --check --status || {
  echo "Piper JSON checksum mismatch" >&2; exit 1;
}

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
Copy the audited Gemma Q4_K_M GGUF and matching projector into $MODELS.
The source repository was not recorded by the previous agent, so this baseline
refuses to invent one. The exact accepted filenames and SHA-256 identities are:
  $GGUF
  $GEMMA_SHA256
  $MMPROJ
  $MMPROJ_SHA256
--------------------------------------------------------------------
EOF
fi
if [ -f "$GGUF" ]; then
  echo "$GEMMA_SHA256  $GGUF" | sha256sum --check --status || {
    echo "Gemma GGUF checksum mismatch" >&2; exit 1;
  }
fi
if [ -f "$MMPROJ" ]; then
  echo "$MMPROJ_SHA256  $MMPROJ" | sha256sum --check --status || {
    echo "Gemma projector checksum mismatch" >&2; exit 1;
  }
fi

echo
echo "Model inventory under $MODELS:"
find "$MODELS" -maxdepth 2 -type f -printf "  %-60p %10s bytes\n" 2>/dev/null || ls -lR "$MODELS"

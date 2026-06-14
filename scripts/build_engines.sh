#!/usr/bin/env bash
# Build TensorRT engines from ONNX, ON THE JETSON (engines are device-specific).
#
#   ./scripts/build_engines.sh
#
# Inputs:  $ARGUS_HOME/exports/*.onnx
# Outputs: $ARGUS_HOME/engines/*.engine
set -euo pipefail

ARGUS_HOME="${ARGUS_HOME:-/opt/argus}"
EXPORTS="$ARGUS_HOME/exports"
ENGINES="$ARGUS_HOME/engines"
mkdir -p "$ENGINES"

# Locate trtexec
TRTEXEC="$(command -v trtexec || true)"
[ -z "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC=/usr/src/tensorrt/bin/trtexec
if [ -z "$TRTEXEC" ]; then
  echo "trtexec not found. It ships with the JetPack TensorRT (/usr/src/tensorrt/bin)."
  exit 1
fi
echo "Using trtexec: $TRTEXEC"

# ---- YOLO-World ----
YOLO_ONNX="$EXPORTS/yoloworld_640.onnx"
if [ -f "$YOLO_ONNX" ]; then
  echo "Building YOLO-World FP16 engine..."
  "$TRTEXEC" \
    --onnx="$YOLO_ONNX" \
    --saveEngine="$ENGINES/yoloworld_640_fp16.engine" \
    --fp16 \
    --memPoolSize=workspace:2048
else
  echo "Skip YOLO-World: $YOLO_ONNX not found."
fi

# ---- RAFT-Stereo (optional) ----
RAFT_ONNX="$EXPORTS/raft_stereo_640x480.onnx"
if [ -f "$RAFT_ONNX" ]; then
  echo "Building RAFT-Stereo FP16 engine..."
  "$TRTEXEC" \
    --onnx="$RAFT_ONNX" \
    --saveEngine="$ENGINES/raft_stereo_fp16.engine" \
    --fp16 \
    --memPoolSize=workspace:2048
else
  echo "Skip RAFT-Stereo: $RAFT_ONNX not found (SGBM depth backend will be used)."
fi

echo "Engines in $ENGINES:"
ls -lh "$ENGINES" || true

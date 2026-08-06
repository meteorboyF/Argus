#!/usr/bin/env bash
# Launch Gemma 4 E2B with vision via the native llama.cpp server.
# Exposes an OpenAI-compatible endpoint on :8080 that accepts image input.
#
#   ./scripts/run_llama_server.sh
#
# The argus runtime talks to this at http://127.0.0.1:8080 (see config agent.server_url).
set -euo pipefail

ARGUS_HOME="${ARGUS_HOME:-/opt/argus}"
LLAMA_BIN="$ARGUS_HOME/llama.cpp/build/bin/llama-server"
MODELS="$ARGUS_HOME/models"

MODEL="$MODELS/gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ="$MODELS/mmproj-gemma4-e2b-f16.gguf"

# The 8 GB Orin Nano cannot load this model/projector while also reserving CUDA
# compute buffers. Keep the default profile CPU-resident; larger Jetsons can
# opt back into GPU offload through these environment variables.
LLAMA_DEVICE="${LLAMA_DEVICE:-none}"
LLAMA_NGL="${LLAMA_NGL:-0}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_FIT="${LLAMA_FIT:-off}"
LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-off}"

if [ ! -x "$LLAMA_BIN" ]; then
  echo "llama-server not found at $LLAMA_BIN — run scripts/setup_jetson.sh first."
  exit 1
fi
for f in "$MODEL" "$MMPROJ"; do
  if [ ! -f "$f" ]; then
    echo "Missing model file: $f"
    echo "Copy the Gemma 4 E2B GGUF + vision projector into $MODELS first."
    exit 1
  fi
done

echo "Starting Gemma 4 E2B (vision) on :8080 ..."
# --jinja enables the model's own chat template (needed for native tool calling
# if agent.tool_protocol=native; harmless for the default prompt protocol).
exec "$LLAMA_BIN" \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --device "$LLAMA_DEVICE" \
  -ngl "$LLAMA_NGL" \
  --parallel "$LLAMA_PARALLEL" \
  --fit "$LLAMA_FIT" \
  --flash-attn "$LLAMA_FLASH_ATTN" \
  --ctx-size 2048 \
  --jinja \
  --no-mmproj-offload \
  --host 127.0.0.1 \
  --port 8080

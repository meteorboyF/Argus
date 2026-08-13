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

# R36.4.7 had an NVIDIA NVMAP allocator regression that made even small CUDA
# allocations fail. The reference Orin Nano was upgraded to R36.5.2 and the
# full decoder offload below was verified with a real privacy-gated image query.
# Keep the large vision projector on CPU (--no-mmproj-offload) for headroom.
LLAMA_DEVICE="${LLAMA_DEVICE:-CUDA0}"
LLAMA_NGL="${LLAMA_NGL:-99}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_FIT="${LLAMA_FIT:-off}"
LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-off}"
# Gemma's chat template runs a hidden "thinking" pass by default (llama.cpp
# logs "chat template, thinking = 1" at load) and it is NOT free: the model's
# reasoning trace is billed against max_tokens same as the visible answer.
# Measured on-device: with thinking on, a 128-token budget was consumed
# entirely by an unfinished reasoning trace (finish_reason=length, content=""
# — the actual answer never started). With --reasoning off, the same request
# answered directly in ~10 completion tokens and ~20-30s wall clock, down from
# 120+ seconds that frequently never completed. ARGUS needs "one or two short
# spoken sentences," not a visible chain of thought — disable it.
LLAMA_REASONING="${LLAMA_REASONING:-off}"
# Prompt caching can otherwise grow toward an 8 GiB default limit. Repeated
# wearable observations do not justify that risk on an 8 GB unified-memory SoC.
LLAMA_CACHE_RAM="${LLAMA_CACHE_RAM:-0}"

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
  --reasoning "$LLAMA_REASONING" \
  --cache-ram "$LLAMA_CACHE_RAM" \
  --ctx-size 2048 \
  --jinja \
  --no-mmproj-offload \
  --host 127.0.0.1 \
  --port 8080

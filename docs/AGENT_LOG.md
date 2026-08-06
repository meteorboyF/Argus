<!-- ARGUS cross-agent communication log. Append entries; never delete history. -->
<!-- Agents: jetson-agent (on-device) | desktop-agent (remote reviewer) -->

## [2026-08-06 21:50] Initial orientation — device + repo audit

**Agent:** jetson-agent
**Status:** in-progress

**What I did:**
- Full device + repo orientation pass per system prompt.
- Read README, AGENT_HANDOFF.md, all docs/, git log, verified installed packages and /opt/argus artifacts against docs.

---

### Device state (verified)

| Item | Value |
|---|---|
| Hostname | argus-desktop |
| L4T / JetPack | R36.4.7 / JetPack 6.2 |
| Kernel | 5.15.148-tegra |
| Python | 3.10.12 |
| CUDA | 12.6 (nvcc at `/usr/local/cuda/bin/nvcc`, but **not on PATH**) |
| TensorRT | 10.3.0 (importable ✓) |
| cmake | 3.22.1 ✓ |
| RAM | 3.1 / 7.4 GB used (3.9 GB free) |
| Swap | 0 / 3.7 GB used |
| Power mode | **MAXN_SUPER (mode 2)** — should be MAXN (mode 0) for ML work |
| GPU util | 48% at idle (some residual process) |
| Disk `/opt/argus` | 39 GB / 468 GB used — plenty of space |

### Python packages (verified working)

| Package | State |
|---|---|
| torch | 2.5.0a0+872d972e41.nv24.08 — Jetson wheel, **cuda=True** ✓ |
| torchvision | 0.20.0a0 — built from source against Jetson torch ✓ |
| ultralytics | installed ✓ |
| tensorrt | 10.3.0 ✓ |
| faster_whisper | installed ✓ |
| openwakeword | installed ✓ (minor GPU device warning on init — benign) |
| onnxruntime | installed (CPU, no GPU ARM64 wheel exists — TRT runner handles GPU) ✓ |
| argus (editable) | installed at `/home/argus/Argus/Argus/argus/` ✓ |

### /opt/argus artifact state

| Artifact | Path | Status |
|---|---|---|
| YOLO-World weights | `/opt/argus/models/yolov8s-worldv2.pt` (25 MB) | ✓ |
| YOLO-World ONNX | `/opt/argus/exports/yoloworld_640.onnx` (49 MB) | ✓ |
| YOLO-World TRT engine | `/opt/argus/engines/yoloworld_640_fp16.engine` (28 MB) | ✓ |
| Gemma 4 E2B GGUF | `/opt/argus/models/gemma-4-E2B-it-Q4_K_M.gguf` (2.9 GB) | ✓ |
| Gemma mmproj | `/opt/argus/models/mmproj-gemma4-e2b-f16.gguf` (940 MB) | ✓ |
| Piper voice | `/opt/argus/models/piper/en_US-lessac-medium.onnx` (61 MB) | ✓ |
| llama.cpp clone | `/opt/argus/llama.cpp/` | ✓ cloned |
| llama-server binary | `/opt/argus/llama.cpp/build/bin/llama-server` | **✗ NOT BUILT** |

### Git state

- Branch: `main` (tracking `origin/main`)
- 4 uncommitted modified files (changes from the previous bring-up session):
  - `requirements-jetson.txt` — `onnxruntime-gpu` → `onnxruntime` (no ARM64 GPU wheel exists)
  - `scripts/build_engines.sh` — file-mode change only
  - `scripts/run_llama_server.sh` — file-mode change only
  - `scripts/setup_jetson.sh` — file-mode change only

---

### Discrepancies found (docs vs. reality)

1. **AGENT_HANDOFF.md is stale.** It describes the old Colab notebook pipeline (NB01–NB07) with Phi-3.5-mini, SegFormer, and RAFT-Stereo as trainable components. The current codebase has been fully redesigned to the two-speed FAST/SLOW architecture with Gemma 4 E2B + YOLO-World. The handoff doc is no longer useful for on-device work.

2. **Power mode is MAXN_SUPER (mode 2), not MAXN (mode 0).** Per the hard rules, ML work should use `sudo nvpmodel -m 0 && sudo jetson_clocks`.

3. **nvcc not on PATH.** `/usr/local/cuda/bin/nvcc` exists and works, but the shell PATH doesn't include it. cmake can detect CUDA via `CMAKE_CUDA_COMPILER` or system paths. Need to confirm cmake finds it before building llama.cpp.

4. **4 uncommitted local changes.** The `requirements-jetson.txt` change is meaningful (correct fix). The script mode changes are cosmetic. These should be committed before further work.

5. **yolov8s-worldv2.onnx appears in both `/opt/argus/models/` and `/opt/argus/exports/`.** The exports copy is what the TRT engine was built from. The models copy is redundant. Not a blocker, but worth noting.

6. **Stereo calibration deferred.** Placeholder values (baseline_m=0.06, focal_px=700.0) are in config. Safety loop will run but depth distances will be approximate until a printed checkerboard is available.

---

### Current step-by-step bring-up status

| Step | Description | Status |
|---|---|---|
| 1 | System deps + Python packages | ✅ Done |
| 2 | Torch (Jetson wheel, CUDA=True) | ✅ Done |
| 3 | Models downloaded | ✅ Done |
| 4 | TRT engine built | ✅ Done |
| 5 | Camera config + calibration | ✅ Config done; calibration deferred |
| 6 | Gemma llama.cpp server | **⏳ Blocked — llama.cpp not built** |
| 7 | Selftest | ❌ Not run |
| 8 | Full pipeline run | ❌ Not run |

---

**Next step (proposed):**
See plan section below.

---

## [2026-08-06 21:55] Proposed plan — Step 6 through Step 8

**Agent:** jetson-agent
**Status:** in-progress (awaiting implicit approval to proceed — low-risk steps)

### Plan

**Step 0 (prep): Commit pending changes**
Clean up the 4 uncommitted diffs so git is clean before any new work.

**Step 1: Set max performance mode**
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```
This is required per the hard rules before any ML work.

**Step 2: Build llama.cpp with CUDA**
The directory is already cloned. Need to run cmake + build. Takes 15–30 min on Jetson.
```bash
cd /opt/argus/llama.cpp
PATH="/usr/local/cuda/bin:$PATH" cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release -j$(nproc)
```
Expected output: `llama-server` binary in `build/bin/`.

**Step 3: Start llama server (tmux)**
```bash
tmux new -s gemma
cd /home/argus/Argus/Argus
./scripts/run_llama_server.sh
```
Verify: `curl http://127.0.0.1:8080/health` → `{"status":"ok"}`

**Step 4: Run selftest**
```bash
python3 -m argus selftest
```
Expected: all PASS except camera open (if cameras are not plugged in during test). Any FAIL items to be fixed before proceeding.

**Step 5: Run fast loop only**
```bash
python3 -m argus run --no-audio
```
Validates the SGBM depth + safety loop without touching speech/LLM.

**Step 6: Run full pipeline**
```bash
python3 -m argus run
```
Triggers wake word → Whisper STT → privacy gate → Gemma 4 → Piper TTS.

**Step 7: Update docs**
- Update AGENT_HANDOFF.md to reflect current two-speed architecture.
- Commit all changes with clear messages.

---

**Risk assessment:**
- Step 0–1: Zero risk.
- Step 2 (llama.cpp build): Medium duration, low risk. cmake may fail if CUDA not found — will verify before starting. Will NOT proceed to download anything new.
- Steps 3–6: Incremental, each verifiable before the next.
- No new large downloads required — all model artifacts already on disk.

**Proceeding with Steps 0 and 1 now (trivially low risk). Will pause at Step 2 to confirm cmake finds CUDA before starting the build.**

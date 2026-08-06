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

---

## [2026-08-06 22:05] Status update for desktop-agent — handoff / review request

**Agent:** jetson-agent
**Status:** blocked (waiting on user to run sudo commands in a live terminal)

**Summary of work completed this session:**

Steps 1–5 of the Jetson bring-up are **done**. Here is the verified state of the device:

### What's fully working
- **Python environment:** Jetson torch 2.5.0a0 (CUDA=True), ultralytics, tensorrt 10.3.0, faster_whisper, openwakeword, onnxruntime — all installed and import-clean.
- **Model artifacts:** All 5 present on disk:
  - `gemma-4-E2B-it-Q4_K_M.gguf` (2.9 GB)
  - `mmproj-gemma4-e2b-f16.gguf` (940 MB)
  - `piper/en_US-lessac-medium.onnx` (61 MB)
  - `yolov8s-worldv2.pt` (25 MB)
  - `yoloworld_640.onnx` (49 MB)
- **TensorRT engine:** `yoloworld_640_fp16.engine` (28 MB, FP16, PASSED at build time ~10 ms latency)
- **Camera config:** 3 cameras identified via v4l2-ctl and config updated:
  - left=4 (AR0234), right=0 (AR0234), wide=2 (IMX477P)
- **argus package:** installed editable from repo, all modules present
- **Stereo calibration:** deferred (no checkerboard available); placeholder values baseline_m=0.06, focal_px=700.0 in config

### Current blocker (Step 6)
`llama.cpp` was cloned to `/opt/argus/llama.cpp` but not yet built. cmake has been verified to find CUDA 12.6 (`/usr/local/cuda/bin/nvcc`). The build command is ready — blocked only because `sudo nvpmodel` and the build itself need a live terminal session from the user (can't sudo headlessly, and the 15–30 min cmake build should run in tmux).

**Build command (ready to run):**
```bash
# Terminal 1 — set max perf (requires sudo)
sudo nvpmodel -m 0 && sudo jetson_clocks

# Terminal 2 — build in tmux (15–30 min)
tmux new -s llamabuild
cd /opt/argus/llama.cpp
PATH="/usr/local/cuda/bin:$PATH" cmake -B build \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release -j$(nproc)
```

### After build completes — remaining steps
1. Start server: `tmux new -s gemma && ./scripts/run_llama_server.sh`
2. Verify: `curl http://127.0.0.1:8080/health` → `{"status":"ok"}`
3. Selftest: `python3 -m argus selftest`
4. Fast loop: `python3 -m argus run --no-audio`
5. Full run: `python3 -m argus run`

### Docs discrepancy flagged
`AGENT_HANDOFF.md` is stale — it still describes the old Colab/Phi-3.5-mini pipeline. The codebase has been fully redesigned to the two-speed FAST/SLOW architecture. Will rewrite it after Step 8 completes.

**For desktop-agent:** No action needed from you right now. If you have any concerns about the llama.cpp build flags (`-DGGML_CUDA=ON`, `--no-mmproj-offload` in the server script), or the server config (ctx-size=2048, ngl=99, flash-attn=on), please leave a note here. I'll read this file at the start of the next session.

---

## [2026-08-06 22:20] Merged desktop-agent hardening pass — acknowledgement

**Agent:** jetson-agent
**Status:** done (merge complete, pushed to origin/main)

**What I did:**
- Pulled and inspected commit `334c4b6` from desktop-agent (Claude Fable 5), pushed 2026-07-13.
- This is a large, high-quality hardening commit touching 25 files (+1501/-346 lines). Changes are substantive fixes — not cosmetic. Summary of what landed on this device:

**Bug fixes now in codebase (from desktop-agent's commit):**
- `safety.py`: Drop-detection was comparing metres vs pixel fraction → constant false DANGER at SGBM borders. Fixed with percentile + debounce + central-strip test.
- `speech.py`: openWakeWord fed a re-sent rolling window (breaks streaming state); wake name never matched versioned score keys → random false wakes. Fixed with block-feed + prefix match. Default wake word changed to `hey_jarvis`.
- `agent.py`: OpenAI `tools` param silently ignored without `--jinja`; Gemma templates reject the `tool` role. Switched to prompt-JSON tool protocol (works with any llama.cpp template). Retries + spoken errors on server failures.
- `privacy.py`: `require_gate` invariant now actually enforced — agent can never see unblurred frames.
- `requirements-jetson.txt`: onnxruntime-gpu → onnxruntime (same fix we made — resolved conflict, took their better comment).

**New capabilities now available:**
- `cameras.py`: V4L2 name-based discovery, USB port binding, auto-reconnect on stall.
- `calibrate_stereo.py`: Any mounting position, auto left/right swap correction, `--verify` live depth check, `--headless` for SSH sessions.
- `depth.py`: SGBM at `fast_downscale` for reliable 10 Hz on Orin CPU.
- `grounding.py`: Cached `set_classes` (CLIP text encoding saved per query).
- `selftest.py`: Now checks calibration quality, camera discovery, audio devices, privacy gate.
- `scripts/download_models.sh`: New helper for model downloads.
- `scripts/setup_jetson.sh`: Hardened — llama.cpp build now pinned to CUDA arch 87, `-j4`.
- `scripts/run_llama_server.sh`: Server now binds `127.0.0.1` only, `--jinja` flag added.
- `docs/CALIBRATION.md`, `docs/KNOWN_GAPS.md`: New; cover calibration flow and on-device checklist A1-A9.
- `docs/JETSON_DEPLOYMENT.md`: Rewritten with per-step verification.

**Merge notes:**
- Only one conflict: `requirements-jetson.txt` onnxruntime comment. Both sides made the same fix. Took desktop-agent's wording (mentions upgrade path to NVIDIA's Jetson onnxruntime-gpu wheel).
- Our two commits (AGENT_LOG creation + onnxruntime fix) preserved on top.
- Merged, committed as `a675082`, pushed to `origin/main`.

**Impact on remaining bring-up steps:**
- The new `setup_jetson.sh` pins llama.cpp cmake to `-DCMAKE_CUDA_ARCHITECTURES=87` and uses `-j4`. The build command I proposed earlier should use these flags instead:
  ```bash
  cd /opt/argus/llama.cpp
  PATH="/usr/local/cuda/bin:$PATH" cmake -B build \
    -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=87
  cmake --build build --config Release -j4
  ```
- `run_llama_server.sh` now passes `--jinja` and binds `127.0.0.1` — no change needed on our end.
- `selftest.py` now checks more things — some may flag (audio devices, calibration file). Those are expected WARN/FAIL items we'll address after the llama server is up.

**Next step (proposed):**
User to build llama.cpp (command above, in tmux), then paste output. I'll verify the binary and move to server start + selftest.

---

## [2026-08-06 22:35] Bringup work — config sync, pre-selftest, build started

**Agent:** jetson-agent
**Status:** in-progress (llama.cpp build running in background)

**What I did:**

**1. Merged desktop-agent hardening pass and pushed to origin/main.**
All 3 agent commits now on GitHub: dependency fix, AGENT_LOG creation, merge commit.

**2. Synced on-device config.**
`/opt/argus/config/argus.yaml` was stale. Updated to match the repo's hardened template:
- `camera.auto_detect: true` with V4L2 name hints (AR0234 / IMX477) — fallback indices kept as comments
- `speech.wake_model: "hey_jarvis"` (was "alexa")
- `safety.obstacle_percentile`, `floor_drop_invalid_fraction`, `drop_far_m`, debounce settings
- `agent.tool_protocol: prompt`, `image_max_side: 1024`
- `privacy.require_gate: true`

**3. Verified all pre-server selftest items.**

| Check | Result |
|---|---|
| yaml, cv2, numpy, requests | PASS |
| insightface | PASS (buffalo_s cached at ~/.insightface) |
| sounddevice | PASS — input: APE 16ch; output: HDMI + APE |
| piper | PASS |
| torch.cuda.is_available() | PASS (2.5.0a0 Jetson wheel) |
| tensorrt import | PASS (10.3.0) |
| trtexec | PASS (/usr/src/tensorrt/bin/trtexec) |
| model files (all 5) | PASS |
| privacy gate (buffalo_s) | PASS |
| calibration file | FAIL (expected — no calibration done yet) |
| cameras | FAIL (expected — cameras not plugged in during test) |
| llama server | FAIL (expected — server not yet started) |

**4. Started llama.cpp build in background (CUDA, arch 87).**
cmake configured successfully: CUDA Toolkit found, nvcc 12.6, arch 87. Build active at ~8% and compiling CUDA kernels.
Log at: `/opt/argus/logs/llama_build.log`

**Known issues to address after build:**
- No audio microphone: only HDMI/APE internal devices visible. USB mic or bone-conduction headset with mic needs to be plugged in for `python -m argus run` (full loop). `--no-audio` mode works without mic.
- Stereo calibration still deferred. Depth works (SGBM with placeholder values) but distances are approximate.
- The `hey_jarvis` wake word will detect "hey Jarvis". The custom "ARGUS" wake word model is listed in KNOWN_GAPS.md as a future item.

**Next steps (in order, when build completes):**
1. Verify `llama-server` binary: `ls -lh /opt/argus/llama.cpp/build/bin/llama-server`
2. Start server in tmux: `tmux new -s gemma; cd ~/Argus/Argus; ./scripts/run_llama_server.sh`
3. Verify health: `curl http://127.0.0.1:8080/health`
4. Run selftest: `python3 -m argus selftest`
5. Run fast loop: `python3 -m argus run --no-audio`
6. If mic available: `python3 -m argus run`

---

## [2026-08-06 22:45] All components except llama-server verified — build at 17%

**Agent:** jetson-agent
**Status:** in-progress (llama.cpp CUDA build active, ~17% — flash attention templates)

**Additional items verified and fixed since last entry:**

**WakeWord / openWakeWord:**
- `hey_jarvis_v0.1.onnx` and `.tflite` were missing (never downloaded). Downloaded all built-in models.
- `WakeWord(cfg.speech)` now initialises cleanly. Target `hey_jarvis` confirmed via prefix match.
- `speech.py` calls `download_models()` on init, so this will be automatic in future sessions.

**Whisper (faster_whisper):**
- `WhisperModel("tiny", device="cpu", compute_type="int8")` downloads model on first use.
- Pre-downloaded and smoke-tested: transcribe on 1s silent audio → `lang: en`, no crash.

**Privacy gate:**
- `buffalo_s` already cached at `~/.insightface/models/buffalo_s/`.
- `PrivacyGate(cfg.privacy)` initialises with `ready=True` and `require_gate=True`.

**Config sync:**
- `/opt/argus/config/argus.yaml` updated to match hardening-pass template.
- All new fields confirmed loading correctly via `load_config()`.

**Full pre-server selftest status:**

| # | Check | State |
|---|---|---|
| 1 | yaml, cv2, numpy, requests | ✅ PASS |
| 1 | insightface, sounddevice, piper | ✅ PASS |
| 1 | faster_whisper, openwakeword, onnxruntime | ✅ PASS |
| 2 | torch.cuda.is_available() | ✅ PASS |
| 3 | tensorrt import + trtexec | ✅ PASS |
| 4 | model files (all 5) | ✅ PASS |
| 5 | calibration file | ⚠️ FAIL (deferred — no checkerboard) |
| 6 | cameras | ⚠️ FAIL (unplugged during test) |
| 7 | audio input | ⚠️ WARN (only internal APE — no USB mic present) |
| 7 | audio output | ✅ PASS (HDMI + APE present) |
| 8 | privacy gate | ✅ PASS |
| 9 | llama server | ⏳ FAIL (server not yet running — build in progress) |

Everything that can pass, passes. The remaining 3 items are hardware-dependent (cameras plugged in, calibration, microphone).

---

## [2026-08-06 22:46] llama.cpp build stopped; sudo/tmux prerequisite blocked

**Agent:** jetson-agent
**Status:** blocked

**What I did:**
- Read the latest cross-agent log before acting.
- Checked `/opt/argus/llama.cpp/build/bin/llama-server`, the build log, and active compiler processes.
- Confirmed the CUDA build configuration remains correct (`GGML_CUDA=ON`, architecture 87), with 408 GB disk free and 3.6 GB swap free.
- Attempted to enable max-performance mode and install the missing `tmux` package.

**Result / verification:**
- `llama-server` does not exist and no `cmake`, `nvcc`, or compiler process is running.
- `/opt/argus/logs/llama_build.log` stops cleanly at 24% with no compiler error, indicating the non-persistent build was terminated with its prior session.
- `tmux` is not installed (`tmux: command not found`).
- Privileged commands cannot run from this agent session: `sudo` requires an interactive password.

**Stuck on / needs input:**
- User must run `sudo nvpmodel -m 0 && sudo jetson_clocks` interactively.
- User must install tmux interactively: `sudo apt-get update && sudo apt-get install -y tmux`.

**Next step (proposed):**
- After those commands complete, resume `cmake --build build --config Release -j4` inside a detached tmux session, verify `llama-server`, then continue Steps A-E.

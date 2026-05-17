# ARGUS — AI-Powered Smart Glasses for the Visually Impaired

Real-time navigation assistant running on a **Jetson Orin Nano Super (8 GB)**.  
Combines stereo depth, semantic segmentation, object detection, face privacy, LLM reasoning, and speech I/O.

---

## Hardware

| Component | Spec |
|-----------|------|
| Compute   | NVIDIA Jetson Orin Nano Super 8 GB |
| Stereo cameras | 2× Arducam AR0234 (USB 3.0, 640×480 @ 60 fps) |
| Wide camera | IMX477P (USB 3.0, 1920×1080 @ 30 fps) |
| Microphone | USB, 16 kHz mono |
| Speaker | 3.5 mm jack |

---

## Model Stack

| Task | Model |
|------|-------|
| Stereo depth | RAFT-Stereo → ONNX → TensorRT |
| Segmentation | SegFormer-B2 (10 ARGUS classes) |
| Object detection | YOLOv8-small + ByteTrack |
| Face detection | SCRFD / insightface buffalo_s |
| Text detection | CRAFT (clovaai/CRAFT-pytorch) |
| LLM | Phi-3.5 Mini Q4\_K\_M (llama-cpp-python, 35 GPU layers) |
| STT | faster-whisper tiny INT8 |
| TTS | Piper en\_US-lessac-medium ONNX |
| Wake word | openWakeWord custom 'ARGUS' |

> **Model weights are NOT stored in this repo.** Download them separately and place under `models/` (see `configs/argus_config.yaml` for expected paths).

---

## Notebook Run Order

Run notebooks in Google Colab (A100 recommended) in this order:

| # | Notebook | Purpose |
|---|----------|---------|
| 00 | `00_setup.ipynb` | Install dependencies, mount Drive |
| 01 | `01_stereo_depth.ipynb` | Train & export RAFT-Stereo |
| 02 | `02_segmentation.ipynb` | Fine-tune SegFormer-B2 |
| 03 | `03_object_detection.ipynb` | Fine-tune YOLOv8-small |
| 04 | `04_privacy_filter.ipynb` | Test face blur + CRAFT text detection |
| 05 | `05_speech_pipeline.ipynb` | Whisper STT + Piper TTS + wake word |
| 06 | `06_llm_setup.ipynb` | Phi-3.5 Mini GGUF inference test |
| 07 | `07_integration_test.ipynb` | Full WorldModelBuilder integration test |

---

## Jetson Deployment

```bash
# 1. Clone repo
git clone https://github.com/meteorboyF/Argus.git
cd Argus/ARGUS

# 2. Install Python deps (JetPack 6 / Python 3.10)
pip install ultralytics transformers faster-whisper piper-tts \
            insightface openwakeword llama-cpp-python \
            opencv-python-headless albumentations

# 3. Copy model weights from Drive → models/
# (see configs/argus_config.yaml for exact paths)

# 4. Stereo camera calibration (once, per physical setup)
python scripts/calibrate_stereo.py --left 0 --right 1 --output configs/stereo_calib.npz

# 5. Run ARGUS
python src/argus_pipeline.py
```

---

## ARGUS Semantic Classes

| ID | Class |
|----|-------|
| 0 | background |
| 1 | navigable\_floor |
| 2 | wall |
| 3 | door\_closed |
| 4 | door\_open |
| 5 | stairs\_up |
| 6 | stairs\_down |
| 7 | person |
| 8 | furniture |
| 9 | clutter |

---

## Privacy

- **Faces** are blurred in-frame with Gaussian blur before any network sees the image.  
- Objects tagged `private=True` in the World Model are never spoken aloud.

---

## Datasets

| Model | Dataset |
|-------|---------|
| SegFormer | ADE20K + NYUv2 + custom staircase/door annotations |
| YOLOv8 | COCO 2017 + custom indoor Roboflow dataset |
| RAFT-Stereo | SceneFlow (pretrain) + indoor AR0234 stereo pairs (fine-tune) |
| Wake word | ~200 custom 'ARGUS' recordings via openWakeWord trainer |

Dataset download links are in `01_stereo_depth.ipynb` and `02_segmentation.ipynb`.

---

## License

MIT

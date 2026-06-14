"""Central configuration for the ARGUS runtime.

Values are loaded from config/argus.yaml when present, with the defaults below
as a fallback. Paths default to the on-device layout (/opt/argus/...) but can be
overridden with the ARGUS_HOME environment variable for local testing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # PyYAML is in requirements; degrade gracefully
    yaml = None


# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
ARGUS_HOME = Path(os.environ.get("ARGUS_HOME", "/opt/argus"))
MODELS_DIR = ARGUS_HOME / "models"
ENGINES_DIR = ARGUS_HOME / "engines"        # TensorRT engines (built on-device)
EXPORTS_DIR = ARGUS_HOME / "exports"        # ONNX produced on PC/Colab
CONFIG_DIR = ARGUS_HOME / "config"
LOGS_DIR = ARGUS_HOME / "logs"


@dataclass
class CameraConfig:
    # Indices are discovered with scripts/calibrate_stereo.py / v4l2-ctl.
    left_index: int = 0           # AR0234 left  (stereo)
    right_index: int = 1          # AR0234 right (stereo)
    wide_index: int = 2           # IMX477P wide (scene / grounding)
    stereo_width: int = 1280
    stereo_height: int = 720
    stereo_fps: int = 30
    wide_width: int = 1920
    wide_height: int = 1080
    wide_fps: int = 30
    calibration_file: str = str(CONFIG_DIR / "stereo_calib.npz")


@dataclass
class DepthConfig:
    # backend: "sgbm" (CPU/OpenCV, always available) or "raft_trt" (TensorRT engine)
    backend: str = "sgbm"
    raft_engine: str = str(ENGINES_DIR / "raft_stereo_fp16.engine")
    min_disparity: int = 0
    num_disparities: int = 128    # must be divisible by 16
    block_size: int = 5
    # Stereo baseline (metres) and focal length (px) — filled by calibration.
    baseline_m: float = 0.06
    focal_px: float = 700.0


@dataclass
class SafetyConfig:
    # Geometric, non-ML reflex. Distances in metres.
    warn_distance_m: float = 1.5      # start warning
    danger_distance_m: float = 0.7    # urgent warning
    floor_drop_threshold_m: float = 0.4   # step-down / hole detection
    roi_bottom_fraction: float = 0.6  # consider lower portion of frame as path
    tick_hz: float = 10.0             # fast-loop frequency


@dataclass
class GroundingConfig:
    # YOLO-World via TensorRT (built on-device from yoloworld_640.onnx).
    engine: str = str(ENGINES_DIR / "yoloworld_640_fp16.engine")
    onnx: str = str(EXPORTS_DIR / "yoloworld_640.onnx")
    weights_pt: str = str(MODELS_DIR / "yolov8s-worldv2.pt")  # ultralytics fallback
    conf_threshold: float = 0.25
    imgsz: int = 640


@dataclass
class AgentConfig:
    # Gemma 4 E2B via native llama.cpp server (OpenAI-compatible endpoint).
    server_url: str = "http://127.0.0.1:8080"
    model_gguf: str = str(MODELS_DIR / "gemma-4-E2B-it-Q4_K_M.gguf")
    mmproj_gguf: str = str(MODELS_DIR / "mmproj-gemma4-e2b-f16.gguf")
    ctx_size: int = 2048
    max_tokens: int = 256
    temperature: float = 0.3
    request_timeout_s: float = 30.0


@dataclass
class SpeechConfig:
    # All speech runs on CPU.
    wake_model: str = "alexa"          # openWakeWord model name or custom .onnx path
    wake_threshold: float = 0.5
    sample_rate: int = 16000
    whisper_model: str = "tiny"        # faster-whisper
    whisper_compute: str = "int8"
    piper_voice: str = str(MODELS_DIR / "piper" / "en_US-lessac-medium.onnx")
    record_seconds: float = 5.0


@dataclass
class PrivacyConfig:
    # Mandatory gate: faces (and optionally text) blurred before the agent sees a frame.
    face_model: str = "buffalo_s"      # insightface SCRFD pack
    det_size: int = 640
    blur_kernel: int = 51
    enable_text_blur: bool = False     # CRAFT — enable once weights are present


@dataclass
class ArgusConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def _merge(dc, overrides: dict):
    """Shallow-merge a dict of overrides into a dataclass instance in place."""
    for key, val in (overrides or {}).items():
        if hasattr(dc, key):
            setattr(dc, key, val)


def load_config(path: str | os.PathLike | None = None) -> ArgusConfig:
    """Load config from YAML, falling back to defaults. Section keys map to the
    dataclasses above (camera, depth, safety, grounding, agent, speech, privacy)."""
    cfg = ArgusConfig()
    path = Path(path) if path else (CONFIG_DIR / "argus.yaml")
    if yaml is not None and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        _merge(cfg.camera, data.get("camera"))
        _merge(cfg.depth, data.get("depth"))
        _merge(cfg.safety, data.get("safety"))
        _merge(cfg.grounding, data.get("grounding"))
        _merge(cfg.agent, data.get("agent"))
        _merge(cfg.speech, data.get("speech"))
        _merge(cfg.privacy, data.get("privacy"))
    return cfg

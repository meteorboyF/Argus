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
    # Smart discovery: when auto_detect is true the rig identifies cameras by
    # V4L2 device name (stereo_name_hint / wide_name_hint) and by resolution
    # grouping, and matches left-vs-right using the USB port paths stored in the
    # calibration file. Indices below are the fallback when discovery cannot
    # decide (and are overwritten in memory once discovery succeeds).
    auto_detect: bool = True
    stereo_name_hint: str = "AR0234"   # substring of the V4L2 name of the stereo cams
    wide_name_hint: str = "IMX477"     # substring of the V4L2 name of the wide cam
    max_probe_index: int = 10          # probe /dev/video0..N during discovery
    left_index: int = 0           # AR0234 left  (stereo) — fallback / override
    right_index: int = 1          # AR0234 right (stereo) — fallback / override
    wide_index: int = 2           # IMX477P wide (scene / grounding) — fallback
    # Capture format and sensor-native dimensions. Per-camera transforms below
    # are applied immediately after capture, before calibration/depth/agent use.
    pixel_format: str = "YUYV"
    stereo_width: int = 960
    stereo_height: int = 600
    stereo_fps: int = 30
    wide_width: int = 1920
    wide_height: int = 1080
    wide_fps: int = 30
    left_rotation: int = 0       # clockwise degrees: 0, 90, 180, or 270
    right_rotation: int = 0
    wide_rotation: int = 0
    left_flip_horizontal: bool = False
    right_flip_horizontal: bool = False
    wide_flip_horizontal: bool = False
    left_flip_vertical: bool = False
    right_flip_vertical: bool = False
    wide_flip_vertical: bool = False
    reconnect: bool = True             # reopen a camera that stops delivering frames
    reconnect_interval_s: float = 2.0
    # Free-running USB cameras have no hardware trigger, so left/right are only
    # approximately simultaneous. Pairs captured further apart than this are
    # dropped rather than fed to depth — during head rotation a large skew
    # produces a confident but wrong depth map. 0 disables the check.
    max_skew_ms: float = 12.0
    calibration_file: str = str(CONFIG_DIR / "stereo_calib.npz")


@dataclass
class DepthConfig:
    # backend: "sgbm" (CPU/OpenCV, always available) or "raft_trt" (TensorRT engine)
    backend: str = "sgbm"
    raft_engine: str = str(ENGINES_DIR / "raft_stereo_fp16.engine")
    min_disparity: int = 0
    num_disparities: int = 128    # must be divisible by 16
    block_size: int = 5
    # SGBM at full stereo resolution is too slow for the 10 Hz fast loop on the
    # Orin Nano CPU. Rectified frames are downscaled by this factor before
    # matching; the disparity is scaled back so metric depth stays correct.
    fast_downscale: int = 2
    # Stereo baseline (metres) and focal length (px) — filled by calibration.
    baseline_m: float = 0.06
    focal_px: float = 700.0
    # Calibration (rectification maps + Q) from scripts/calibrate_stereo.py.
    # When present, depth is rectified and metric for the actual mounting geometry.
    calibration_file: str = str(CONFIG_DIR / "stereo_calib.npz")
    # Calibration drift watch. After rectification, matched features must share a
    # row; a growing vertical residual means a camera has moved and depth has
    # silently gone wrong. Sampled every health_interval_s, not every frame.
    # The calibrator targets < ~1 px vertical error, so 2 px is clearly degraded.
    health_check: bool = True
    health_interval_s: float = 5.0
    health_max_vertical_px: float = 2.0
    health_min_matches: int = 25       # fewer than this = textureless scene, no verdict
    health_consecutive: int = 3        # debounce, same idea as the drop-off reflex
    health_downscale: int = 2
    health_max_features: int = 400


@dataclass
class SafetyConfig:
    # Geometric, non-ML reflex. Distances in metres.
    warn_distance_m: float = 1.5      # start warning
    danger_distance_m: float = 0.7    # urgent warning
    # Obstacle distance = this percentile of valid depths in the path ROI
    # (a raw min is a single-pixel measurement and fires on speckle noise).
    obstacle_percentile: float = 5.0
    min_valid_pixels: int = 200       # need at least this many finite depths to judge
    # Floor drop-off: fraction of the bottom strip (central columns) that is
    # invalid-or-far before we call it a drop. SGBM leaves a band of invalid
    # pixels on the left edge, so only the central strip is considered.
    floor_drop_invalid_fraction: float = 0.75
    drop_far_m: float = 3.0           # bottom-strip depth beyond this counts as "floor fell away"
    drop_consecutive_ticks: int = 3   # debounce: require N ticks in a row
    roi_bottom_fraction: float = 0.6  # consider lower portion of frame as path
    tick_hz: float = 10.0             # fast-loop frequency
    danger_repeat_s: float = 2.0      # min seconds between spoken DANGER warnings
    warn_repeat_s: float = 6.0        # min seconds between spoken WARN notices


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
    # Gemma reasoning model via native llama.cpp server (OpenAI-compatible endpoint).
    server_url: str = "http://127.0.0.1:8080"
    model_gguf: str = str(MODELS_DIR / "gemma-4-E2B-it-Q4_K_M.gguf")
    mmproj_gguf: str = str(MODELS_DIR / "mmproj-gemma4-e2b-f16.gguf")
    ctx_size: int = 2048
    max_tokens: int = 48
    temperature: float = 0.3
    request_timeout_s: float = 90.0
    retries: int = 0                   # avoid doubling a slow on-device timeout
    # Tool calling: "prompt" (default — model returns a JSON action in plain text,
    # works with every llama.cpp chat template) or "native" (OpenAI tools param;
    # requires llama-server started with --jinja and a template that supports it).
    # Native tool_calls in the response are honoured in both modes.
    tool_protocol: str = "prompt"
    # Downscale the gated frame to this max side before base64-encoding it for
    # the VLM — keeps prompt processing time and memory sane on the Jetson.
    image_max_side: int = 256


@dataclass
class SpeechConfig:
    # All speech runs on CPU.
    wake_model: str = "hey_jarvis"     # openWakeWord model name or custom .onnx path
    wake_threshold: float = 0.5
    sample_rate: int = 16000
    whisper_model: str = "tiny"        # faster-whisper
    whisper_compute: str = "int8"
    piper_voice: str = str(MODELS_DIR / "piper" / "en_US-lessac-medium.onnx")
    record_seconds: float = 5.0
    # sounddevice selectors: None = system default. Accepts an integer index or
    # a name substring (e.g. "USB"). List devices with: python -m sounddevice
    input_device: int | str | None = None
    output_device: int | str | None = None


@dataclass
class PrivacyConfig:
    # Mandatory gate: faces (and optionally text) blurred before the agent sees a frame.
    face_model: str = "buffalo_s"      # insightface SCRFD pack
    det_size: int = 640
    blur_kernel: int = 51
    enable_text_blur: bool = False     # CRAFT — enable once weights are present
    # Hard enforcement: when true (production), the agent is NEVER called if the
    # face detector failed to initialise. Set false only for bench debugging.
    require_gate: bool = True


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

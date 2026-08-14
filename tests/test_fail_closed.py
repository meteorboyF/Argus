from types import SimpleNamespace

import numpy as np
import pytest

from argus.config import DepthConfig, GroundingConfig
from argus.depth import DepthEstimator
from argus.grounding import Detection, Grounder
from argus.orchestrator import Orchestrator


def test_production_depth_refuses_missing_gpu_engine(tmp_path):
    cfg = DepthConfig(
        backend="raft_trt",
        raft_engine=str(tmp_path / "missing.engine"),
        allow_cpu_fallback=False,
        health_check=False,
    )
    with pytest.raises(RuntimeError, match="GPU depth engine is missing"):
        DepthEstimator(cfg)


def test_cpu_depth_requires_explicit_diagnostic_override():
    cfg = DepthConfig(backend="sgbm", allow_cpu_fallback=False, health_check=False)
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        DepthEstimator(cfg)


def test_production_grounding_refuses_missing_trt_engine(tmp_path):
    cfg = GroundingConfig(
        backend="trt",
        engine=str(tmp_path / "missing.engine"),
        allow_torch_fallback=False,
    )
    with pytest.raises(RuntimeError, match="TensorRT engine is missing"):
        Grounder(cfg)


def test_torch_grounding_requires_explicit_diagnostic_override():
    cfg = GroundingConfig(backend="torch", allow_torch_fallback=False)
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        Grounder(cfg)


def test_trt_grounding_postprocesses_runtime_label(monkeypatch):
    grounder = Grounder.__new__(Grounder)
    grounder.cfg = GroundingConfig(conf_threshold=0.25, imgsz=640)
    output = np.zeros((1, 5, 8400), dtype=np.float32)
    output[0, :, 10] = [320, 320, 160, 80, 0.9]
    grounder._runner = SimpleNamespace(
        infer=lambda feeds: {"output0": output})
    grounder._embed = lambda name: np.ones((1, 1, 512), dtype=np.float32)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detection = grounder._find_object_trt("keys", frame)

    assert detection is not None
    assert detection.name == "keys"
    assert detection.confidence == pytest.approx(0.9)
    assert detection.bbox == (240, 200, 400, 280)


def test_trt_grounding_does_not_guess_below_threshold():
    grounder = Grounder.__new__(Grounder)
    grounder.cfg = GroundingConfig(conf_threshold=0.25, imgsz=640)
    grounder._runner = SimpleNamespace(
        infer=lambda feeds: {"output0": np.zeros((1, 5, 8400), dtype=np.float32)})
    grounder._embed = lambda name: np.ones((1, 1, 512), dtype=np.float32)
    assert grounder._find_object_trt(
        "keys", np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_uncalibrated_depth_cannot_reach_safety_rules():
    orch = Orchestrator.__new__(Orchestrator)
    orch.depth = SimpleNamespace(calibrated=False)
    orch._reported_uncalibrated = False
    orch.safety = SimpleNamespace(
        evaluate=lambda _: pytest.fail("uncalibrated depth reached safety evaluation"))
    assert orch._evaluate_safety_depth(np.ones((8, 8), dtype=np.float32)) is None


def test_privacy_exception_cancels_image_query():
    spoken = []
    orch = Orchestrator.__new__(Orchestrator)
    orch.privacy = SimpleNamespace(
        apply=lambda _: (_ for _ in ()).throw(RuntimeError("detector failure")))
    orch.speaker = SimpleNamespace(speak=spoken.append)
    assert orch._apply_privacy(np.zeros((4, 4, 3), dtype=np.uint8)) is None
    assert spoken and "Privacy filter failed" in spoken[0]


def test_unverified_cross_camera_projection_omits_distance():
    orch = Orchestrator.__new__(Orchestrator)
    det = Detection("keys", 0.9, (60, 20, 80, 40), (70, 30))
    result = orch._fuse_detection("keys", det, np.zeros((100, 100, 3), dtype=np.uint8))
    assert result["found"] is True
    assert result["distance_m"] is None
    assert result["distance_verified"] is False
    assert result["direction"] == "right"

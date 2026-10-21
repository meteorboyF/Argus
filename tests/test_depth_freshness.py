import threading
import time
from types import SimpleNamespace

import numpy as np
from argus.depth_freshness import depth_is_current
from argus.orchestrator import Orchestrator


def test_depth_age_uses_capture_time_not_gpu_completion():
    assert depth_is_current(1, 1.25)
    assert not depth_is_current(1, 1.251)
    assert not depth_is_current(2, 1)
    assert not depth_is_current(None, 1)


def test_stale_gpu_result_cannot_heartbeat_or_reach_hazard_rules():
    o = Orchestrator.__new__(Orchestrator)
    o.cfg = SimpleNamespace(safety=SimpleNamespace(tick_hz=10))
    o._stop = threading.Event()
    def capture():
        o._stop.set()  # one tick only
        return SimpleNamespace(left=np.zeros((2,2,3)), right=np.zeros((2,2,3)),
                               ts=time.perf_counter()-1, skew_ms=0)
    o.rig = SimpleNamespace(get_stereo_pair=capture)
    o.depth = SimpleNamespace(calibrated=True, depth_map=lambda *args: np.ones((2,2)))
    o._depth_lock = threading.Lock()
    o._latest_depth = np.ones((2,2))
    o._latest_depth_ts = time.perf_counter()-1
    o._fast_error = None
    o._fast_ready = threading.Event()
    o._skew_ok = lambda pair: True
    o._check_calibration_health = lambda: None
    o._note_tick = lambda dt: None
    # No speaker/preview/rule engine: reaching them fails the test.
    o._fast_loop()
    assert o._fast_error is None
    assert not o._fast_ready.is_set()
    assert o.latest_depth() is None
    from argus.telemetry import bus
    assert bus.get("safety_level") == "UNKNOWN"


def test_cached_depth_expires_even_when_camera_never_recovers():
    o = Orchestrator.__new__(Orchestrator)
    o._depth_lock = threading.Lock()
    o._latest_depth = np.ones((2,2))
    o._latest_depth_ts = time.perf_counter()
    assert o.latest_depth() is not None
    o._latest_depth_ts -= 1
    assert o.latest_depth() is None


def test_slow_mover_detection_cannot_announce_old_geometry():
    from argus.safety import Level
    o = Orchestrator.__new__(Orchestrator)
    o.cfg = SimpleNamespace(safety=SimpleNamespace(tick_hz=10, danger_repeat_s=0, warn_repeat_s=0))
    o._stop = threading.Event()
    pair=SimpleNamespace(left=np.zeros((2,2,3)), right=np.zeros((2,2,3)),
                         ts=time.perf_counter(), skew_ms=0)
    def capture():
        pair.ts=time.perf_counter()
        o._stop.set()
        return pair
    o.rig=SimpleNamespace(get_stereo_pair=capture)
    o.depth=SimpleNamespace(calibrated=True, depth_map=lambda *args: np.ones((2,2)))
    o._depth_lock=threading.Lock()
    o._fast_error=None
    o._fast_ready=threading.Event()
    o._last_danger=o._last_warn=0
    o._skew_ok=lambda pair: True
    o._publish_depth=lambda depth: None
    o._update_slam_and_nav=lambda *args: None
    o._evaluate_safety_depth=lambda depth: SimpleNamespace(level=Level.DANGER, message='Stop')
    alert=SimpleNamespace(key='head', level='DANGER', text='Stop')
    o._corridor_alerts=lambda *args: [alert]
    def slow_detector(*args):
        pair.ts-=1  # simulate source becoming stale during shared-GPU detection
        return [alert]
    o._mover_alerts=slow_detector
    o._check_calibration_health=lambda: None
    o._note_tick=lambda dt: None
    # No speaker exists: announcing either alert would fail the loop.
    o._fast_loop()
    assert o._fast_error is None
    from argus.telemetry import bus
    assert bus.get('safety_level') == 'UNKNOWN'

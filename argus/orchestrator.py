"""Two-speed orchestration — the ARGUS nervous system.

Fast loop (thread, always on):
    stereo capture -> depth -> SafetyReflex -> speak urgent hazards immediately.

Slow loop (main thread, event-driven):
    wake word -> record -> transcribe -> wide frame -> PRIVACY GATE -> Gemma
    -> [optional find_object via YOLO-World + depth fusion] -> Piper speaks.

The privacy gate is a hard precondition: with privacy.require_gate=true (the
production default) the agent is never called unless the gate initialised, and
never on a frame that has not passed through it.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .agent import AgentError, GemmaAgent
from .calib_health import CalibState
from .cameras import CameraRig
from .config import ArgusConfig
from .depth import DepthEstimator
from .grounding import Grounder
from .privacy import PrivacyGate
from .safety import Level, SafetyReflex
from .speech import Priority, Speaker, Transcriber, WakeWord, record


def _direction_of(col: int, width: int) -> str:
    if col < width * 0.35:
        return "left"
    if col > width * 0.65:
        return "right"
    return "center"


class Orchestrator:
    def __init__(self, cfg: ArgusConfig, enable_audio: bool = True):
        self.cfg = cfg
        # Validate required production inference backends before opening cameras
        # or starting worker threads. A fail-closed startup must not leak devices.
        self.grounder = Grounder(cfg.grounding)
        self.depth = DepthEstimator(cfg.depth)
        self.privacy = PrivacyGate(cfg.privacy)
        if cfg.privacy.require_gate and not self.privacy.ready:
            self.depth.health.stop()
            raise RuntimeError(
                "Privacy gate failed to initialise and privacy.require_gate is true. "
                "The agent must never see unblurred frames — fix insightface/onnxruntime "
                "(see selftest), or set privacy.require_gate: false for bench debugging only.")

        self.rig = CameraRig(cfg.camera)
        self.safety = SafetyReflex(cfg.safety)
        self.agent = GemmaAgent(cfg.agent)
        self.speaker = Speaker(cfg.speech, enabled=enable_audio)
        self.enable_audio = enable_audio

        self._latest_depth: np.ndarray | None = None
        self._depth_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_danger = 0.0
        self._last_warn = 0.0
        self._skew_drops = 0
        self._last_skew_report = 0.0
        self._tick_count = 0
        self._tick_time = 0.0
        self._last_calib_state = CalibState.UNKNOWN
        self._reported_uncalibrated = False

        # Wake word + STT are created lazily in _listen_loop so `argus query`
        # and --no-audio runs don't pay their load time (or need a microphone).
        self.wake: WakeWord | None = None
        self.stt: Transcriber | None = None

    # ------------------------------------------------------------------ fast loop
    def _fast_loop(self):
        period = 1.0 / self.cfg.safety.tick_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            pair = self.rig.get_stereo_pair()
            if pair is not None and self._skew_ok(pair):
                depth_m = self.depth.depth_map(pair.left, pair.right)
                with self._depth_lock:
                    self._latest_depth = depth_m
                # Uncalibrated disparity cannot support a metric danger decision.
                # Keep producing diagnostic depth, but never turn it into wearable
                # hazard speech until physical calibration has been verified.
                state = self._evaluate_safety_depth(depth_m)
                now = time.perf_counter()
                # DANGER speaks immediately (rate-limited); WARN speaks on a
                # longer cadence so the user isn't flooded. Both hand off to the
                # speaker thread and return at once — this loop must never block
                # on audio, or it stops watching for hazards while it talks.
                if state is not None and state.level == Level.DANGER and (now - self._last_danger) > self.cfg.safety.danger_repeat_s:
                    self.speaker.speak(state.message, Priority.DANGER)
                    self._last_danger = now
                elif state is not None and state.level == Level.WARN and (now - self._last_warn) > self.cfg.safety.warn_repeat_s:
                    self.speaker.speak(state.message, Priority.WARN)
                    self._last_warn = now
            self._check_calibration_health()
            dt = time.perf_counter() - t0
            self._note_tick(dt)
            time.sleep(max(0.0, period - dt))

    def _evaluate_safety_depth(self, depth_m: np.ndarray):
        """Only calibrated metric depth may drive wearable hazard speech."""
        if not self.depth.calibrated:
            if not self._reported_uncalibrated:
                print("[safety] stereo is uncalibrated; metric/danger speech suppressed")
                self._reported_uncalibrated = True
            return None
        return self.safety.evaluate(depth_m)

    def _check_calibration_health(self):
        """Announce calibration drift once, when it is first detected.

        Deliberately does not stop the safety loop: degraded warnings still beat
        no warnings, and a blind user mid-street is worse off with the system
        silent. But they must be told the distances have stopped being reliable,
        because otherwise the failure is invisible — bad depth reads exactly like
        good depth.
        """
        state = self.depth.health.state
        if state is self._last_calib_state:
            return
        self._last_calib_state = state
        if state is CalibState.DEGRADED:
            self.speaker.speak(
                "Warning. My distance sensing has drifted and may be inaccurate. "
                "Please re-calibrate.", Priority.WARN)

    def _skew_ok(self, pair) -> bool:
        """Reject stereo pairs whose two frames are too far apart in time.

        The AR0234s are free-running USB cameras with no hardware trigger, so
        left and right are only ever approximately simultaneous. Global shutter
        removes motion skew *within* a frame but not the offset *between* the
        two. While the user turns their head, that offset shifts disparity
        across the whole image and SGBM happily returns a confident, wrong depth
        map — which becomes a wrong safety decision with no visible symptom.
        Dropping the pair is safe; trusting it is not.
        """
        limit = self.cfg.camera.max_skew_ms
        if limit <= 0 or pair.skew_ms <= limit:
            return True
        self._skew_drops += 1
        now = time.perf_counter()
        if now - self._last_skew_report > 5.0:
            print(f"[cameras] dropped {self._skew_drops} stereo pairs over the "
                  f"{limit:.0f} ms skew limit (latest {pair.skew_ms:.1f} ms)")
            self._last_skew_report = now
            self._skew_drops = 0
        return False

    def _note_tick(self, dt: float):
        """Track achieved fast-loop rate and complain if it falls behind.

        REQ-NF01 assumes this loop actually runs at tick_hz. If depth starts
        taking longer than the period the loop silently slows down, warnings
        arrive late, and nothing anywhere says so — so say so.
        """
        self._tick_count += 1
        self._tick_time += dt
        if self._tick_count < 100:
            return
        achieved = self._tick_count / self._tick_time if self._tick_time else 0.0
        target = self.cfg.safety.tick_hz
        if achieved < target * 0.8:
            print(f"[safety] fast loop running at {achieved:.1f} Hz, target {target:.1f} Hz "
                  f"— hazard warnings are late; lower depth cost (fast_downscale) or tick_hz")
        self._tick_count = 0
        self._tick_time = 0.0

    def latest_depth(self) -> np.ndarray | None:
        with self._depth_lock:
            return None if self._latest_depth is None else self._latest_depth.copy()

    # ------------------------------------------------------------------ slow loop
    def handle_query(self, question: str):
        """Run one full slow-path interaction for an already-transcribed query."""
        frame = self.rig.get_wide_frame()
        if frame is None:
            self.speaker.speak("Camera not ready.")
            return

        # HARD PRECONDITION: privacy gate before the agent sees anything.
        if self.cfg.privacy.require_gate and not self.privacy.ready:
            self.speaker.speak("Privacy filter unavailable. I can't answer right now.")
            return
        gated = self._apply_privacy(frame)
        if gated is None:
            return

        try:
            reply = self.agent.ask(gated, question)
            if reply.tool_call == "find_object":
                name = (reply.tool_args or {}).get("name", "")
                det = self.grounder.find_object(name, gated) if name else None
                tool_result = self._fuse_detection(name, det, gated)
                reply = self.agent.with_tool_result(question, tool_result)
        except AgentError as e:
            print(f"[agent] {e}")
            self.speaker.speak("Sorry, my reasoning engine is not responding.")
            return

        self.speaker.speak(reply.text or "I'm not sure.")

    def _apply_privacy(self, frame: np.ndarray) -> np.ndarray | None:
        """Fail closed: a privacy exception cancels the image query."""
        try:
            gated, _ = self.privacy.apply(frame)
            return gated
        except Exception as e:  # noqa: BLE001 — privacy failure is a safe refusal
            print(f"[privacy] gate failed; image query cancelled: {e}")
            self.speaker.speak("Privacy filter failed. I can't answer that image question.")
            return None

    def _fuse_detection(self, name: str, det, frame) -> dict:
        """Return a grounded direction without inventing cross-camera distance.

        Wide-to-stereo intrinsics/extrinsics are not calibrated. Proportional
        pixel scaling was geometrically invalid, so distance remains absent until
        a verified projection is implemented.
        """
        if det is None:
            return {"found": False, "name": name}
        cx, cy = det.center
        return {
            "found": True,
            "name": name,
            "confidence": round(det.confidence, 2),
            "direction": _direction_of(cx, frame.shape[1]),
            "distance_m": None,
            "distance_verified": False,
        }

    # ------------------------------------------------------------------ lifecycle
    def start_fast_loop(self):
        """Start the safety fast loop in a background thread (used by `run` and
        by one-shot `argus query` so depth fusion has data)."""
        threading.Thread(target=self._fast_loop, daemon=True).start()

    def run(self):
        self.start_fast_loop()
        self.speaker.speak("ARGUS ready.")
        try:
            if not self.enable_audio:
                # Headless: keep the fast loop running; slow loop driven externally.
                while not self._stop.is_set():
                    time.sleep(0.5)
                return
            self._listen_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _listen_loop(self):
        from .speech import mic_stream
        if self.wake is None:
            self.wake = WakeWord(self.cfg.speech)
        if self.stt is None:
            self.stt = Transcriber(self.cfg.speech)
        sr = self.cfg.speech.sample_rate
        # openWakeWord keeps its own streaming state — feed each 80 ms block
        # exactly once (see speech.py docstring).
        for block in mic_stream(sr, device=self.cfg.speech.input_device):
            if self._stop.is_set():
                break
            if self.wake.detected(block):
                self.speaker.speak("Yes?")
                audio = record(self.cfg.speech.record_seconds, sr,
                               device=self.cfg.speech.input_device)
                question = self.stt.transcribe(audio)
                if question:
                    self.handle_query(question)
                # speak() is non-blocking now, so wait for the answer to finish
                # before listening again — otherwise the mic hears ARGUS talking
                # and the user gets answered over. The fast loop is unaffected;
                # it has its own thread and never waits here.
                self.speaker.wait_until_idle(timeout=30.0)
                self.wake.reset()

    def stop(self):
        self._stop.set()
        self.speaker.stop()
        self.depth.health.stop()
        self.rig.release()

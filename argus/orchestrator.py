"""Two-speed orchestration — the ARGUS nervous system.

Fast loop (thread, always on):
    stereo capture -> depth -> SafetyReflex -> speak urgent hazards immediately.

Slow loop (main thread, event-driven):
    wake word -> record -> transcribe -> wide frame -> PRIVACY GATE -> Gemma 4
    -> [optional find_object via YOLO-World + depth fusion] -> Piper speaks.

The privacy gate is a hard precondition: the agent is never called on a frame
that has not passed through it.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .agent import GemmaAgent
from .cameras import CameraRig
from .config import ArgusConfig
from .depth import DepthEstimator
from .grounding import Grounder
from .privacy import PrivacyGate
from .safety import Level, SafetyReflex
from .speech import Speaker, Transcriber, WakeWord, record


def _direction_of(col: int, width: int) -> str:
    if col < width * 0.35:
        return "left"
    if col > width * 0.65:
        return "right"
    return "center"


class Orchestrator:
    def __init__(self, cfg: ArgusConfig, enable_audio: bool = True):
        self.cfg = cfg
        self.rig = CameraRig(cfg.camera)
        self.depth = DepthEstimator(cfg.depth)
        self.safety = SafetyReflex(cfg.safety)
        self.privacy = PrivacyGate(cfg.privacy)
        self.grounder = Grounder(cfg.grounding)
        self.agent = GemmaAgent(cfg.agent)
        self.speaker = Speaker(cfg.speech, enabled=enable_audio)
        self.enable_audio = enable_audio

        self._latest_depth: np.ndarray | None = None
        self._depth_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_warned = 0.0

        if enable_audio:
            self.wake = WakeWord(cfg.speech)
            self.stt = Transcriber(cfg.speech)

    # ------------------------------------------------------------------ fast loop
    def _fast_loop(self):
        period = 1.0 / self.cfg.safety.tick_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            pair = self.rig.get_stereo_pair()
            if pair is not None:
                depth_m = self.depth.depth_map(pair.left, pair.right)
                with self._depth_lock:
                    self._latest_depth = depth_m
                state = self.safety.evaluate(depth_m)
                # Voice only DANGER, and rate-limit so we don't talk over ourselves.
                if state.level == Level.DANGER and (time.perf_counter() - self._last_warned) > 2.0:
                    self.speaker.speak(state.message)
                    self._last_warned = time.perf_counter()
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))

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
        gated, n_faces = self.privacy.apply(frame)

        reply = self.agent.ask(gated, question)

        if reply.tool_call == "find_object":
            name = (reply.tool_args or {}).get("name", "")
            det = self.grounder.find_object(name, gated)
            tool_result = self._fuse_detection(name, det, gated)
            reply = self.agent.with_tool_result(question, tool_result)

        self.speaker.speak(reply.text or "I'm not sure.")

    def _fuse_detection(self, name: str, det, frame) -> dict:
        """Combine a YOLO-World box with the latest depth map -> 3D-ish position."""
        if det is None:
            return {"found": False, "name": name}
        cx, cy = det.center
        depth_m = self.latest_depth()
        dist = None
        if depth_m is not None:
            # depth map is at stereo resolution; sample proportionally.
            dh, dw = depth_m.shape[:2]
            fh, fw = frame.shape[:2]
            sx, sy = int(cx * dw / fw), int(cy * dh / fh)
            patch = depth_m[max(0, sy - 5):sy + 5, max(0, sx - 5):sx + 5]
            finite = patch[np.isfinite(patch)]
            if finite.size:
                dist = round(float(np.median(finite)), 2)
        return {
            "found": True,
            "name": name,
            "confidence": round(det.confidence, 2),
            "direction": _direction_of(cx, frame.shape[1]),
            "distance_m": dist,
        }

    # ------------------------------------------------------------------ lifecycle
    def run(self):
        fast = threading.Thread(target=self._fast_loop, daemon=True)
        fast.start()
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
        sr = self.cfg.speech.sample_rate
        buf = np.zeros(sr, dtype=np.int16)  # rolling 1s window for wake detection
        for block in mic_stream(sr):
            if self._stop.is_set():
                break
            buf = np.concatenate([buf, block])[-sr:]
            if self.wake.detected(buf):
                self.speaker.speak("Yes?")
                audio = record(self.cfg.speech.record_seconds, sr)
                question = self.stt.transcribe(audio)
                if question:
                    self.handle_query(question)
                buf[:] = 0  # reset window after handling

    def stop(self):
        self._stop.set()
        self.rig.release()

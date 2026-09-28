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

from .agent import AgentError, GemmaAgent, requested_object
from .calib_health import CalibState
from .cameras import CameraRig
from .config import ArgusConfig
from .depth import DepthEstimator
from .grounding import Grounder
from .privacy import PrivacyGate
from .safety import Level, SafetyReflex
from .speech import Priority, Speaker, Transcriber, WakeWord, record
from .telemetry import bus


def _direction_of(col: int, width: int) -> str:
    if col < width * 0.35:
        return "left"
    if col > width * 0.65:
        return "right"
    return "center"


class Orchestrator:
    def __init__(self, cfg: ArgusConfig, enable_audio: bool = True,
                 enable_mic: bool | None = None):
        self.cfg = cfg
        self.enable_mic = enable_audio if enable_mic is None else enable_mic
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
        self._fast_ready = threading.Event()
        self._fast_error: Exception | None = None
        self._fast_thread: threading.Thread | None = None
        self._query_lock = threading.Lock()
        self._dashboard = None
        self.slam = None
        self.navigator = None
        intr = self.depth.intrinsics()
        if intr is not None:
            if cfg.slam.enabled:
                from .slam import StereoVisualOdometry
                self.slam = StereoVisualOdometry(cfg.slam, intr["fx"], intr["fy"],
                                                 intr["cx"], intr["cy"], intr["baseline_m"])
            if cfg.navigation.enabled:
                from .navigation import Navigator
                self.navigator = Navigator(cfg.navigation, intr["fx"], intr["cx"])
        bus.set(slam="off (needs stereo calibration)" if self.slam is None else "starting")
        bus.set(depth_backend=self.depth.backend,
                calibration="loaded" if self.depth.calibrated else "absent",
                calibrated=self.depth.calibrated)
        # The CLIP text encoder dominated cold grounding at 11-15 s. Load it
        # while the fast loop starts so the first spoken "find X" is warm.
        threading.Thread(target=self._warm_grounding, name="clip-warmup",
                         daemon=True).start()

        # Wake word + STT are created lazily in _listen_loop so `argus query`
        # and --no-audio runs don't pay their load time (or need a microphone).
        self.wake: WakeWord | None = None
        self.stt: Transcriber | None = None

    def _warm_grounding(self):
        started = time.perf_counter()
        try:
            bus.set(grounding="warming text encoder")
            self.grounder.warm()
            bus.set(grounding=f"ready ({time.perf_counter() - started:.1f}s warm-up)")
        except Exception as exc:  # noqa: BLE001 — warm-up is an optimisation only
            bus.set(grounding=f"warm-up failed: {exc}")

    # ------------------------------------------------------------------ fast loop
    def _fast_loop(self):
        period = 1.0 / self.cfg.safety.tick_hz
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                pair = self.rig.get_stereo_pair()
                if pair is not None:
                    bus.frame("left", pair.left)
                    bus.frame("right", pair.right)
                    bus.set(skew_ms=pair.skew_ms)
                if pair is not None and self._skew_ok(pair):
                    depth_started = time.perf_counter()
                    depth_m = self.depth.depth_map(pair.left, pair.right)
                    bus.set(depth_ms=(time.perf_counter() - depth_started) * 1000.0)
                    with self._depth_lock:
                        self._latest_depth = depth_m
                    self._fast_ready.set()
                    self._publish_depth(depth_m)
                    self._update_slam_and_nav(depth_m, pair.ts)
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
                if pair is None:
                    continue
        except Exception as exc:  # noqa: BLE001
            self._fast_error = exc
            self._fast_ready.set()
            self._stop.set()
            print(f"[safety] fast loop failed closed: {exc}")

    def _update_slam_and_nav(self, depth_m: np.ndarray, ts: float):
        """Pose and corridor guidance; both exist only with calibration."""
        if self.slam is not None and self.depth.last_rectified is not None:
            self._slam_tick = getattr(self, "_slam_tick", 0) + 1
            if self._slam_tick % max(1, self.cfg.slam.submit_every) == 0:
                self.slam.submit(*self.depth.last_rectified, ts)
            st = self.slam.state
            bus.set(slam=(f"{st.reason} {st.quality * 100:.0f}% inliers ({st.inliers}/{st.features}) "
                          f"{st.rate_hz:.1f} Hz  {st.speed_mps:.2f} m/s"),
                    slam_trail=st.trail, slam_yaw_deg=st.yaw_deg, ego_speed_mps=st.speed_mps)
        if self.navigator is not None:
            guidance = self.navigator.evaluate(depth_m)
            bus.set(corridor=guidance.corridor, nav_heading=guidance.heading_deg,
                    nav_status=f"{guidance.status} ({guidance.width_deg:.0f} deg free)")
            phrase = self.navigator.spoken(guidance)
            if phrase:
                self.speaker.speak(phrase, Priority.NORMAL)

    def _publish_depth(self, depth_m: np.ndarray):
        """Dashboard view of the depth map, sampled at most ~10 Hz."""
        from .dashboard import depth_to_color, disparity_to_color
        now = time.perf_counter()
        if now - getattr(self, "_last_depth_publish", 0.0) < 0.1:
            return
        self._last_depth_publish = now
        small = depth_m[::2, ::2]
        if self.depth.calibrated:
            bus.frame("depth_color", depth_to_color(small))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                disp = (self.cfg.depth.focal_px * self.cfg.depth.baseline_m) / small
            disp[~np.isfinite(small)] = 0
            bus.frame("depth_color", disparity_to_color(disp))
        bus.set(fps_left=self.rig.left.fps, fps_right=self.rig.right.fps,
                fps_wide=self.rig.wide.fps,
                link_left=self.rig.left.link_mbps, link_right=self.rig.right.link_mbps,
                link_wide=self.rig.wide.link_mbps)
        wide, _, _ = self.rig.wide.latest()
        bus.frame("wide", wide)

    def _evaluate_safety_depth(self, depth_m: np.ndarray):
        """Only calibrated metric depth may drive wearable hazard speech."""
        if not self.depth.calibrated:
            if not self._reported_uncalibrated:
                print("[safety] stereo is uncalibrated; metric/danger speech suppressed")
                bus.log("system", "stereo uncalibrated: hazard speech suppressed until "
                                  "scripts/calibrate_stereo.py has run")
                self._reported_uncalibrated = True
            bus.set(safety_level="UNCALIBRATED")
            return None
        state = self.safety.evaluate(depth_m)
        bus.set(safety_level=state.level.name, nearest_m=state.min_distance_m,
                nearest_dir=state.direction,
                approach=(f"{state.approach_direction} {state.closing_speed_mps:.1f} m/s "
                          f"TTC {state.ttc_s:.1f} s" if state.approach_direction else "none"))
        return state

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
        bus.set(skew_drops=self._skew_drops)
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
        bus.set(fast_hz=min(achieved, target))
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
        with self._query_lock:
            bus.log("user", question)
            try:
                self._handle_query(question)
            except Exception as exc:  # noqa: BLE001 — a query must never kill the process
                bus.log("system", f"query failed: {exc}")
                self.speaker.speak("Sorry, something went wrong with that question.")
            finally:
                bus.set(gemma="idle")

    def _handle_query(self, question: str):
        query_started = time.perf_counter()
        frame = self.rig.get_wide_frame()
        if frame is None:
            self.speaker.speak("Camera not ready.")
            return

        # HARD PRECONDITION: privacy gate before the agent sees anything.
        if self.cfg.privacy.require_gate and not self.privacy.ready:
            self.speaker.speak("Privacy filter unavailable. I can't answer right now.")
            return
        stage_started = time.perf_counter()
        gated = self._apply_privacy(frame)
        privacy_s = time.perf_counter() - stage_started
        if gated is None:
            return
        bus.frame("wide_gated", gated)
        bus.set(gated_at=time.monotonic())

        try:
            visual_agent_s = 0.0
            forced_name = requested_object(question)
            if forced_name and self.cfg.agent.fast_locate:
                # An explicit "find X" never needs Gemma to decide whether to
                # ground: the policy is deterministic (see DECISION_LOG), so the
                # first ~3 s visual turn is skipped entirely.
                from .agent import AgentReply
                reply = AgentReply(text="", tool_call="find_object",
                                   tool_args={"name": forced_name})
            else:
                stage_started = time.perf_counter()
                bus.set(gemma="looking at the frame")
                reply = self.agent.ask(gated, question)
                visual_agent_s = time.perf_counter() - stage_started
                if forced_name and reply.tool_call != "find_object":
                    print("[agent] explicit locate request bypassed grounding; forcing find_object")
                    reply.tool_call = "find_object"
                    reply.tool_args = {"name": forced_name}
                    reply.text = ""
            if reply.tool_call == "find_object":
                name = (reply.tool_args or {}).get("name", "")
                stage_started = time.perf_counter()
                bus.set(grounding=f"looking for '{name}'")
                det = self.grounder.find_object(name, gated) if name else None
                grounding_s = time.perf_counter() - stage_started
                bus.detections([det] if det is not None else [])
                bus.set(grounding=f"'{name}': {'found' if det else 'not found'} "
                                  f"({grounding_s * 1000:.0f} ms)")
                tool_result = self._fuse_detection(name, det, gated)
                stage_started = time.perf_counter()
                bus.set(gemma="phrasing the answer")
                reply = self.agent.with_tool_result(question, tool_result)
                final_agent_s = time.perf_counter() - stage_started
                total_s = time.perf_counter() - query_started
                print(
                    f"[query] privacy={privacy_s:.3f}s visual_agent={visual_agent_s:.3f}s "
                    f"grounding={grounding_s:.3f}s final_agent={final_agent_s:.3f}s "
                    f"total={total_s:.3f}s "
                    f"found={tool_result.get('found')} "
                    f"direction={tool_result.get('direction')} distance_omitted="
                    f"{tool_result.get('distance_m') is None}")
                bus.set(query_latency=f"{total_s:.1f}s (privacy {privacy_s:.2f}, gemma "
                                      f"{visual_agent_s + final_agent_s:.1f}, ground {grounding_s:.2f})")
            else:
                total_s = time.perf_counter() - query_started
                bus.set(query_latency=f"{total_s:.1f}s (privacy {privacy_s:.2f}, gemma {visual_agent_s:.1f})")
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
    def start_fast_loop(self, timeout: float = 8.0):
        """Start the safety fast loop in a background thread (used by `run` and
        by one-shot `argus query` so depth fusion has data)."""
        if self._fast_thread is not None:
            return
        self._fast_thread = threading.Thread(target=self._fast_loop, daemon=True)
        self._fast_thread.start()
        if not self._fast_ready.wait(timeout):
            self._stop.set()
            raise RuntimeError("GPU depth fast loop did not produce a valid stereo result")
        if self._fast_error is not None:
            raise RuntimeError(f"GPU depth fast loop failed: {self._fast_error}") from self._fast_error

    def _ask_file_loop(self, path: str):
        """Accept typed questions appended to a text file (scripted demos, SSH).

        Each new line is one question. The file is created if missing and only
        lines written after startup are used, so old demos never replay.
        """
        import os
        try:
            open(path, "a").close()
            with open(path, "r") as fh:
                fh.seek(0, os.SEEK_END)
                while not self._stop.is_set():
                    line = fh.readline()
                    if not line:
                        time.sleep(0.2)
                        continue
                    question = line.strip()
                    if question:
                        self.handle_query(question)
        except OSError as exc:
            bus.log("system", f"ask file unavailable: {exc}")

    def run(self, dashboard: bool = False, fullscreen: bool = True,
            ask_file: str | None = None):
        from .telemetry import TegraStats
        stats = TegraStats()
        stats.start()
        if ask_file:
            threading.Thread(target=self._ask_file_loop, args=(ask_file,),
                             name="ask-file", daemon=True).start()
            bus.log("system", f"questions can be appended to {ask_file}")
        if dashboard:
            from .dashboard import Dashboard
            self._dashboard = Dashboard(on_question=self.handle_query, fullscreen=fullscreen)
            self._dashboard.start()
        self.start_fast_loop()
        self.speaker.speak("ARGUS ready.")
        try:
            if not self.enable_mic:
                # No wake word / STT: questions arrive typed from the dashboard
                # (or not at all in headless mode). The fast loop keeps running.
                while not self._stop.is_set():
                    if self._dashboard is not None and self._dashboard.closed.is_set():
                        break
                    time.sleep(0.2)
                return
            self._listen_loop()
        except KeyboardInterrupt:
            pass
        finally:
            stats.stop()
            if self._dashboard is not None:
                self._dashboard.stop()
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
            if self._dashboard is not None and self._dashboard.closed.is_set():
                break
            if self.wake.detected(block):
                self.speaker.speak("Yes?")
                bus.set(gemma="listening")
                audio = record(self.cfg.speech.record_seconds, sr,
                               device=self.cfg.speech.input_device,
                               stop_on_silence=self.cfg.speech.stop_on_silence)
                bus.set(gemma="transcribing")
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
        self.rig.release()
        if (self._fast_thread is not None and self._fast_thread.is_alive()
                and self._fast_thread is not threading.current_thread()):
            self._fast_thread.join(timeout=2.0)
        self.depth.health.stop()
        if self.slam is not None:
            self.slam.stop()
        self.speaker.stop()

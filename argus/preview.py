"""On-monitor diagnostic preview for the normalized three-camera rig."""
from __future__ import annotations

import textwrap
import threading
import time

import cv2
import numpy as np

from .agent import GemmaAgent
from .cameras import CameraRig
from .config import ArgusConfig
from .privacy import PrivacyGate


def _pane(frame: np.ndarray, label: str, width: int = 400, height: int = 500) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.rectangle(out, (0, 0), (width, 38), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (80, 255, 80), 2, cv2.LINE_AA)
    return out


def _overlay(frame: np.ndarray, message: str) -> np.ndarray:
    out = frame.copy()
    lines = textwrap.wrap(message, width=100) or [""]
    box_h = 18 + 30 * len(lines)
    shade = out.copy()
    cv2.rectangle(shade, (0, out.shape[0] - box_h),
                  (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.72, out, 0.28, 0, out)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (16, out.shape[0] - box_h + 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_preview(cfg: ArgusConfig, seconds: float = 0, describe: bool = False) -> None:
    """Show upright left/right/wide feeds; optionally describe one gated wide frame."""
    window = "ARGUS camera diagnostic — Q/Esc to close"
    state = {"message": "ARGUS camera preview", "started": False, "done": not describe}
    deadline = time.monotonic() + seconds if seconds > 0 else None

    gate = PrivacyGate(cfg.privacy) if describe else None
    if describe and cfg.privacy.require_gate and not gate.ready:
        raise RuntimeError("Privacy gate is required but unavailable")

    def describe_frame(frame: np.ndarray) -> None:
        try:
            gated, faces = gate.apply(frame)
            started = time.monotonic()
            reply = GemmaAgent(cfg.agent).ask(
                gated, "What is in front of me? Answer in one short sentence.")
            answer = reply.text or "I could not describe the scene."
            state["message"] = (f"ARGUS says: {answer} "
                                f"({time.monotonic() - started:.1f}s, {faces} face(s) blurred)")
            state["done"] = True
            print(state["message"], flush=True)
        except Exception as exc:  # keep the diagnostic window alive on server errors
            state["message"] = f"ARGUS vision error: {exc}"
            state["done"] = True
            print(state["message"], flush=True)

    with CameraRig(cfg.camera) as rig:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1200, 650)
        while (deadline is None or time.monotonic() < deadline
               or (describe and state["started"] and not state["done"])):
            pair = rig.get_stereo_pair()
            wide = rig.get_wide_frame()
            if pair is None or wide is None:
                if cv2.waitKey(10) & 0xFF in (27, ord("q"), ord("Q")):
                    break
                continue
            if describe and not state["started"]:
                state["started"] = True
                state["message"] = "ARGUS is analyzing the privacy-gated center frame..."
                threading.Thread(target=describe_frame, args=(wide.copy(),), daemon=True).start()
            view = np.hstack([
                _pane(pair.left, "LEFT AR0234"),
                _pane(pair.right, "RIGHT AR0234"),
                _pane(wide, "CENTER B0459"),
            ])
            cv2.imshow(window, _overlay(view, state["message"]))
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
        cv2.destroyAllWindows()

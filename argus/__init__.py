"""ARGUS — AI-powered smart glasses runtime for the visually impaired.

Two-Speed Vision-Language architecture for the NVIDIA Jetson Orin Nano Super:
  - Fast path (always on, non-ML): stereo depth + geometric safety reflex
  - Slow path (event-driven): wake word -> STT -> privacy gate -> Gemma 4 E2B
    agent -> YOLO-World grounding tool -> depth fusion -> Piper TTS

See docs/ARCHITECTURE.md for the full design.
"""

__version__ = "0.1.0"

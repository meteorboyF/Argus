"""Speech pipeline — wake word detection, Whisper STT, Piper TTS, and LLM query."""
import time
import json
import wave
import struct
import threading
import numpy as np


ARGUS_SYSTEM = (
    "You are ARGUS, an AI assistant embedded in smart glasses for a blind person. "
    "Rules: "
    "1. SHORT responses — 1-2 sentences max. "
    "2. NEVER describe private=true objects. "
    "3. Safety hazards FIRST. "
    "4. Distances and directions clearly. "
    "5. Speak naturally."
)


def load_wake_word_model(tflite_path: str):
    """Load openWakeWord model for the custom 'ARGUS' wake word."""
    from openwakeword.model import Model  # type: ignore
    model = Model(wakeword_models=[tflite_path], inference_framework='tflite')
    return model


def load_whisper(model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
    """Load faster-whisper model for offline STT."""
    from faster_whisper import WhisperModel  # type: ignore
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def load_tts(model_path: str, config_path: str):
    """
    Load Piper TTS model (en_US-lessac-medium ONNX).

    model_path  — path to .onnx file
    config_path — path to .onnx.json config file
    """
    from piper import PiperVoice  # type: ignore
    voice = PiperVoice.load(model_path, config_path=config_path)
    return voice


def transcribe(whisper_model, audio_path: str) -> str:
    """Run Whisper STT on a WAV file, return transcript string."""
    segments, _ = whisper_model.transcribe(audio_path, beam_size=5, language="en")
    return " ".join(s.text.strip() for s in segments)


def speak(tts_voice, text: str, output_path: str = "/tmp/argus_tts.wav"):
    """Synthesise text to a WAV file using Piper and return the file path."""
    with wave.open(output_path, "w") as wav_file:
        tts_voice.synthesize(text, wav_file)
    return output_path


def world_model_to_speech(world_model: dict, llm, user_query: str = None) -> tuple:
    """
    Build a prompt from the World Model dict, query the local LLM, return (text, latency_ms).

    llm — llama_cpp.Llama instance.
    """
    objects = world_model.get('objects', [])
    hazard  = world_model.get('hazard')
    floor   = world_model.get('navigable_floor', True)
    nearest = world_model.get('nearest_obstacle_dist', 999)

    visible = [o for o in objects if not o.get('private', False)]

    scene_str = "SCENE DATA:\n"
    if hazard:
        scene_str += f"  HAZARD: {hazard}\n"
    if not floor:
        scene_str += "  No clear floor detected\n"
    scene_str += f"  Nearest obstacle: {nearest:.1f}m\n"
    for obj in visible[:8]:
        scene_str += (f"  [Object: {obj['label']}, "
                      f"Distance: {obj['distance']:.1f}m, "
                      f"Direction: {obj['direction']}]\n")

    if user_query:
        user_msg = f"{scene_str}\nUser asks: {user_query}"
    else:
        user_msg = f"{scene_str}\nProvide brief navigation guidance."

    prompt = f"<|system|>\n{ARGUS_SYSTEM}<|end|>\n<|user|>\n{user_msg}<|end|>\n<|assistant|>\n"

    t0  = time.time()
    out = llm(
        prompt,
        max_tokens  = 80,
        temperature = 0.1,
        stop        = ['<|end|>', '<|user|>', '\n\n'],
        echo        = False,
    )
    latency_ms = (time.time() - t0) * 1000
    response   = out['choices'][0]['text'].strip()
    return response, latency_ms

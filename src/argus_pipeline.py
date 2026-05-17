"""
ARGUS main pipeline — stereo depth + wide-camera perception loop.

Hardware:
  2x Arducam AR0234  (USB 3.0, 640x480)  — stereo pair, indices 0 & 1
  1x IMX477P wide    (USB 3.0, 1920x1080) — wide FOV perception, index 2
"""
import os, time, cv2, yaml, torch, numpy as np
from pathlib import Path

from world_model    import WorldModelBuilder
from navigation     import generate_navigation_guidance
from privacy_filter import blur_faces
from speech         import load_wake_word_model, load_whisper, load_tts, world_model_to_speech, speak

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "argus_config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

BASE        = CFG["base_dir"]
DEVICE      = CFG["device"]
STEREO_L    = CFG["cameras"]["stereo_left_index"]
STEREO_R    = CFG["cameras"]["stereo_right_index"]
WIDE_IDX    = CFG["cameras"]["wide_index"]
STEREO_W    = CFG["cameras"]["stereo_width"]
STEREO_H    = CFG["cameras"]["stereo_height"]
WIDE_W      = CFG["cameras"]["wide_width"]
WIDE_H      = CFG["cameras"]["wide_height"]
FPS_TARGET  = CFG["pipeline"]["fps"]
SPEAK_EVERY = CFG["pipeline"]["speak_every_n_frames"]


def load_models():
    from ultralytics import YOLO                          # type: ignore
    from transformers import SegformerForSemanticSegmentation  # type: ignore
    from insightface.app import FaceAnalysis              # type: ignore
    from llama_cpp import Llama                           # type: ignore

    detector  = YOLO(os.path.join(BASE, CFG["models"]["yolo"]))
    segmentor = SegformerForSemanticSegmentation.from_pretrained(
        os.path.join(BASE, CFG["models"]["segformer"])
    ).to(DEVICE).eval()
    face_app  = FaceAnalysis(name="buffalo_s",
                              root=os.path.join(BASE, CFG["models"]["insightface_root"]))
    face_app.prepare(ctx_id=0 if DEVICE != "cpu" else -1, det_size=(320, 320))

    llm = Llama(
        model_path  = os.path.join(BASE, CFG["models"]["phi3_gguf"]),
        n_gpu_layers= CFG["llm"]["n_gpu_layers"],
        n_ctx       = CFG["llm"]["n_ctx"],
        verbose     = False,
    )

    whisper   = load_whisper(CFG["speech"]["whisper_size"],
                              CFG["speech"]["whisper_device"],
                              CFG["speech"]["whisper_compute"])
    tts_voice = load_tts(os.path.join(BASE, CFG["speech"]["tts_model"]),
                          os.path.join(BASE, CFG["speech"]["tts_config"]))
    wake_word = load_wake_word_model(os.path.join(BASE, CFG["speech"]["wakeword_model"]))

    return detector, segmentor, face_app, llm, whisper, tts_voice, wake_word


def open_cameras():
    cap_l = cv2.VideoCapture(STEREO_L, cv2.CAP_V4L2)
    cap_r = cv2.VideoCapture(STEREO_R, cv2.CAP_V4L2)
    cap_w = cv2.VideoCapture(WIDE_IDX, cv2.CAP_V4L2)
    for cap, w, h in [(cap_l, STEREO_W, STEREO_H), (cap_r, STEREO_W, STEREO_H),
                       (cap_w, WIDE_W, WIDE_H)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap_l, cap_r, cap_w


def main():
    print("Loading models…")
    detector, segmentor, face_app, llm, whisper, tts_voice, wake_word = load_models()

    builder = WorldModelBuilder(detector, segmentor, face_app, DEVICE)

    print("Opening cameras…")
    cap_l, cap_r, cap_w = open_cameras()

    frame_idx = 0
    world     = {}

    print("ARGUS running. Press Ctrl+C to stop.")
    try:
        while True:
            t_start = time.time()

            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()
            ok_w, frame_w = cap_w.read()

            if not (ok_l and ok_r and ok_w):
                print("Camera read failure — retrying…")
                time.sleep(0.1)
                continue

            # Depth from stereo (placeholder — replace with RAFT-Stereo / ONNX TRT)
            depth_map = None  # np.ndarray (H, W) in metres when available

            # Wide-camera perception
            frame_w_priv, _ = blur_faces(frame_w, face_app)
            world = builder.process_frame(frame_w_priv, depth_map)

            # Spoken guidance every N frames
            if frame_idx % SPEAK_EVERY == 0:
                guidance = generate_navigation_guidance(world)
                if guidance:
                    wav = speak(tts_voice, guidance)
                    os.system(f"aplay -q {wav} &")

            frame_idx += 1
            elapsed = time.time() - t_start
            sleep   = max(0.0, 1.0 / FPS_TARGET - elapsed)
            time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        cap_l.release()
        cap_r.release()
        cap_w.release()


if __name__ == "__main__":
    main()

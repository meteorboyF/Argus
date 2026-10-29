#!/usr/bin/env python3
"""Interactive private OCR comparison; reports contain metrics, never text/images.

SPACE compares native live left/wide frames. H compares left live with a
12 MP wide still. Hold the same title steady in both views. Q/Esc closes.
Reference is read from a private file; CER includes all recognized text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from argus.acceptance import cer
from argus.cameras import CameraRig
from argus.config import load_config
from argus.ocr import OCREngine


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-file', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    reference = args.reference_file.read_text().strip() if args.reference_file else None
    if reference == '':
        raise ValueError('Reference must be nonempty')
    cv2.setNumThreads(1)
    engine = OCREngine()
    # Warm identical OCR path before measuring either camera.
    engine.read_robust(np.full((600, 960, 3), 255, np.uint8))
    rig = CameraRig(load_config().camera)
    window = 'OCR comparison - SPACE live / H high-res / Q exit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1440, 900)
    busy = threading.Event()
    results = []
    display = {'left': [], 'wide': []}
    message = 'Hold title steady in BOTH views. SPACE: compare live. H: wide 12 MP. Q: exit.'
    lock = threading.Lock()
    worker = None

    def compare(high_res):
        nonlocal message, display
        try:
            capture_start = time.perf_counter()
            if high_res:
                right = rig.capture_still()
                left, lt, _ = rig.left.latest()
                left = left.copy() if left is not None else None
                # Still has no source timestamp; do not claim simultaneous capture.
                skew_ms = None
                valid = left is not None and 0 <= time.perf_counter() - lt <= .5
            else:
                left, lt, _ = rig.left.latest()
                right, rt, _ = rig.wide.latest()
                now = time.perf_counter()
                valid = (left is not None and right is not None
                         and 0 <= now-lt <= .5 and 0 <= now-rt <= .5
                         and abs(lt-rt) <= .1)
                skew_ms = abs(lt-rt)*1000
                left = left.copy() if left is not None else None
                right = right.copy() if right is not None else None
            if not valid or right is None:
                raise RuntimeError('Missing/stale camera frame; wait and retry')
            trial = {'trial': len(results)+1, 'wide_mode': '12MP still' if high_res else 'live',
                     'capture_s': time.perf_counter()-capture_start,
                     'arrival_skew_ms': skew_ms, 'cameras': {}}
            readings = {}
            # Alternate order to reduce consistent first-camera timing bias.
            frames = {'left': left, 'wide': right}
            order = ('left', 'wide') if len(results) % 2 == 0 else ('wide', 'left')
            for name in order:
                reading = engine.read_robust(frames[name])
                words = [w for line in reading.lines for w in line.words]
                readings[name] = reading.masked_text.splitlines()
                trial['cameras'][name] = {
                    'shape': list(frames[name].shape), 'ocr_s': reading.elapsed_s,
                    'lines': len(reading.lines), 'characters': len(reading.text),
                    'mean_confidence': float(np.mean([w.score for w in words])) if words else None,
                    'cer': cer(reference, reading.text) if reference else None}
            with lock:
                results.append(trial)
                display = readings
                message = f'Trial {len(results)} complete. Compare text below. SPACE/H: repeat. Q: exit.'
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({
                    'kind': 'reading-camera-comparison', 'trials': results,
                    'note': 'CER uses exact supplied reference and all recognized text. '
                            'Confidence is not accuracy. No automatic camera selection. '
                            'Live and still modes differ in resolution/capture latency.'}, indent=2))
            print(json.dumps(trial), flush=True)
        except Exception as exc:
            with lock:
                message = f'Comparison failed: {type(exc).__name__}. Wait and retry.'
            print(f'Comparison failed: {type(exc).__name__}: {exc}', flush=True)
        finally:
            busy.clear()

    try:
        while True:
            canvas = np.zeros((900, 1440, 3), np.uint8)
            with lock:
                status, texts = message, dict(display)
                latest = results[-1] if results else None
            for col, name in enumerate(('left', 'wide')):
                x = col*720
                reader = getattr(rig, name)
                frame, ts, _ = reader.latest()
                fresh = frame is not None and 0 <= time.perf_counter()-ts <= .5
                if fresh:
                    scale = min(700/frame.shape[1], 360/frame.shape[0])
                    small = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
                    canvas[45:45+small.shape[0], x+10:x+10+small.shape[1]] = small
                label = name.upper() + (' - LIVE' if fresh else ' - WAITING FOR FRAME')
                cv2.putText(canvas, label, (x+10, 30), 0, .7, (100,255,100), 2)
                y = 440
                if latest:
                    m = latest['cameras'][name]
                    line = f"{m['shape'][1]}x{m['shape'][0]} | OCR {m['ocr_s']:.2f}s"
                    if m['cer'] is not None:
                        line += f" | CER {m['cer']*100:.1f}%"
                    cv2.putText(canvas, line, (x+10,y), 0, .55, (80,220,255), 1)
                    y += 30
                for line in texts[name]:
                    for wrapped in textwrap.wrap(line, 65):
                        if y > 830:
                            break
                        cv2.putText(canvas, wrapped, (x+10,y), 0, .5, (255,255,255), 1)
                        y += 24
            cv2.putText(canvas, status[:145], (10,875), 0, .53, (100,255,255), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 255
            if key in (ord('q'), 27):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if key in (32, ord('h')) and not busy.is_set():
                busy.set()
                with lock:
                    message = 'Capturing and running OCR - keep page still...'
                worker = threading.Thread(target=compare, args=(key == ord('h'),))
                worker.start()
    finally:
        if worker is not None:
            worker.join()
        rig.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Guided, privacy-gated 30-frame collection; run later with Fardeen present.

No raw frame is written. Stop ARGUS before running: this owns the camera rig.
Review the gated images locally before committing any study asset.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from argus.cameras import CameraRig
from argus.config import load_config
from argus.guide import Guide
from argus.ocr import OCREngine
from argus.privacy import PrivacyGate
from argus.sensitive import blur_boxes


def prepare(frame, gate, ocr):
    if not gate.ready:
        raise RuntimeError('Privacy gate is unavailable')
    safe, faces = gate.apply(frame)
    reading = ocr.read(safe)
    return blur_boxes(safe, reading.sensitive_boxes()), faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('reports/bakeoff/frames'))
    parser.add_argument('--tasks', type=Path, default=Path('reports/bakeoff/tasks_t7.json'))
    parser.add_argument('--hold-seconds', type=float, default=8)
    args = parser.parse_args()
    if args.hold_seconds < 3:
        parser.error('Allow at least three seconds to frame each sample')
    cfg = load_config()
    gate, ocr = PrivacyGate(cfg.privacy), OCREngine()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    with Guide(cfg.speech, title='ARGUS model comparison frames') as guide:
        try:
            if not gate.ready:
                raise RuntimeError('Privacy gate is unavailable')
            with CameraRig(cfg.camera) as rig:
                for category, instruction, prompt in (
                    ('scene', 'Aim at a different room scene, away from the monitor.', 'Describe the scene in two short sentences.'),
                    ('sign', 'Hold a different sign or label in clear light.', 'Read the visible label verbatim.'),
                    ('page', 'Hold a different page, including Bangla samples, in clear light.', 'Read the visible page verbatim.')):
                    for i in range(10):
                        name = f'{category}_{i+1:02d}.jpg'
                        path = args.out_dir/name
                        if path.exists():
                            raise RuntimeError('Output frame already exists; choose a new output directory')
                        guide.say(f'{category} {i+1} of 10. {instruction}', force=True)
                        deadline = time.monotonic()+args.hold_seconds
                        while time.monotonic() < deadline or guide.speaking:
                            if guide.show(progress=len(tasks)/30) == 27:
                                raise KeyboardInterrupt
                            time.sleep(.05)
                        frame = rig.get_wide_frame()
                        if frame is None:
                            raise RuntimeError('Wide camera has no frame')
                        safe, faces = prepare(frame, gate, ocr)
                        if not cv2.imwrite(str(path), safe):
                            raise RuntimeError('Could not save gated frame')
                        tasks.append({'id':f'{category}_{i+1:02d}', 'image':name,
                                      'prompt':prompt, 'tokens':280, 'max_tokens':128,
                                      'faces_blurred':faces})
                        args.tasks.parent.mkdir(parents=True, exist_ok=True)
                        args.tasks.write_text(json.dumps(tasks, indent=2)+'\n')
                        guide.say(f'Saved {len(tasks)} of 30 privacy processed frames.', force=True)
            guide.done('Thirty frames collected. Review them locally before sharing. Model comparison is still pending.')
        except (Exception, KeyboardInterrupt) as exc:
            guide.failed(str(exc) or 'Collection cancelled; already saved gated frames remain on disk.')
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

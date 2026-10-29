#!/usr/bin/env python3
"""Render design previews from recorded camera footage; no hardware access.

Transcript, OCR text and state are illustrative layout content. Source camera
frames come from saved demos; these PNGs are not fresh feature-test evidence.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from argus.recording_layout import RecordingLayout
from argus.telemetry import Snapshot,Event


def camera_crop(path,seconds):
    cap=cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError(f'Cannot read preview source: {path}')
        # The old recording contained a 720x405 camera stage and an overlaid
        # label at its top. Remove the old label only for this design mockup.
        return frame[32:405,:720].copy()
    finally:
        cap.release()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recordings-dir',type=Path,default=Path('reports/feature-recordings/2026-10-29'))
    ap.add_argument('--out-dir',type=Path,default=Path('reports/interface-redesign'))
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    now=time.monotonic()
    examples=[
        ('reading','04-reading-page-agent-take-01.mp4',35,'left',
         'Argus, read this page.','Captured. Reading now.'),
        ('conversation','05-conversation-take-01.mp4',8,'wide',
         'Argus, what can you help me with?',
         'I can help you read, find objects and understand your surroundings.'),
        ('privacy','06-privacy-face-blur-take-01.mp4',15,'wide',
         'Argus, what is in front of me?','A person is sitting in the room.'),
    ]
    rendered=[]
    for feature,filename,seconds,source,user,response in examples:
        if not (args.recordings_dir/filename).exists():
            print(f'Skipping unavailable source: {filename}')
            continue
        frame=camera_crop(args.recordings_dir/filename,seconds)
        values={'calibrated':False,'speaking':response if feature=='conversation' else None}
        if feature=='reading':
            values.update(reading_ocr_status='Illustrative text · design preview',
                          reading_ocr_text='Does Structure Matter? An Empirical\nStudy of Data Flow Graphs in\nTransformers for Decompiled Android\nBytecode')
        snap=Snapshot({},values,[Event(now,'user',user),Event(now,'argus',response)],[])
        image=RecordingLayout(feature).render(snap,frame,source=source,
                                               privacy_ready=True,face_count=None)
        cv2.imwrite(str(args.out_dir/f'{feature}-preview.png'),image)
        rendered.append(feature)
    (args.out_dir/'preview-notes.json').write_text(json.dumps({
        'kind':'recording-interface-design-previews',
        'note':'Illustrative transcript/OCR states, recorded camera crops. Not live test evidence.',
        'recordings_dir':str(args.recordings_dir),'rendered':rendered},indent=2))
    print(f'Previews saved to {args.out_dir}')


if __name__=='__main__':
    main()

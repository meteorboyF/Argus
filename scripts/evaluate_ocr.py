#!/usr/bin/env python3
"""Evaluate private OCR samples. Output contains CER/timing, never page text.

Input: JSON list of {image: absolute path, reference: text}. Keep it outside git.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from argus.acceptance import cer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--backend', choices=['rapidocr','tesseract-ben'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.backend == 'rapidocr':
        from argus.ocr import OCREngine
        engine = OCREngine()
    else:
        from argus.bangla_ocr import BanglaOCR
        engine = BanglaOCR()
    samples = []
    for i, sample in enumerate(json.loads(args.manifest.read_text())):
        image = cv2.imread(str(Path(sample['image'])))
        if image is None:
            raise RuntimeError(f'Unreadable sample {i+1}')
        reading = engine.read(image)
        samples.append({'sample':i+1, 'cer':cer(sample['reference'], reading.text), 'ocr_s':reading.elapsed_s})
    import numpy as np
    result = {'status':'partial', 'backend':args.backend, 'samples':samples,
              'median_s':float(np.median([s['ocr_s'] for s in samples])) if samples else None,
              'p95_s':float(np.percentile([s['ocr_s'] for s in samples],95)) if samples else None,
              'note':'Offline OCR evaluation only; capture-to-first-speech and resource evidence remain pending.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

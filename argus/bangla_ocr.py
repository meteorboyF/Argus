"""Optional local Tesseract Bangla evaluator. No implicit package installation.

This adapter intentionally remains outside live reading until measured against
Bangla pages. It never sends page text to a language model or logs it.
"""
import csv
import io
import subprocess
import time

import cv2
import numpy as np
from .ocr import Reading, Word, order_lines
from .sensitive import find_sensitive


class BanglaOCR:
    def __init__(self, languages='ben+eng', timeout_s=30):
        available = subprocess.run(['tesseract', '--list-langs'], capture_output=True,
                                   text=True, check=True, timeout=5).stdout.splitlines()
        missing = set(languages.split('+'))-set(available)
        if missing:
            raise RuntimeError('Missing Tesseract languages: '+', '.join(sorted(missing)))
        self.languages, self.timeout_s = languages, timeout_s

    def read(self, image):
        started = time.perf_counter()
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError('Cannot encode OCR image')
        result = subprocess.run(['tesseract', 'stdin', 'stdout', '-l', self.languages, 'tsv'],
                                input=encoded.tobytes(), capture_output=True, check=True,
                                timeout=self.timeout_s, env=self._environment())
        words = []
        for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8')), delimiter='\t'):
            if not row['text'].strip() or float(row['conf']) < 50:
                continue
            x, y, w, h = (int(row[k]) for k in ('left', 'top', 'width', 'height'))
            words.append(Word(row['text'], np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.float32), float(row['conf'])/100))
        lines = order_lines(words)
        return Reading(lines, time.perf_counter()-started, bool(find_sensitive('\n'.join(l.text for l in lines))))

    @staticmethod
    def _environment():
        import os
        return dict(os.environ, OMP_THREAD_LIMIT='1')

"""The diagnostic must display actual masked OCR and handle missing frames."""
import numpy as np

from argus.demo_view import reading_preview
from argus.telemetry import Snapshot


def test_reading_preview_masks_sensitive_text_and_labels_sources(monkeypatch):
    drawn = []
    monkeypatch.setattr('argus.demo_view._text', lambda img, s, *a, **kw: drawn.append(s))
    frame = np.full((60, 96, 3), 100, np.uint8)
    snap = Snapshot({'reading_capture': frame},
                    {'reading_capture_at': 1, 'reading_ocr_text': 'Your OTP is 482913.',
                     'reading_ocr_status': 'OCR completed'}, [], [])
    image = reading_preview(frame, snap, 'left')
    assert image.shape == (900, 1440, 3)
    text = '\n'.join(drawn)
    assert 'LEFT' in text and 'LAST CAPTURE' in text
    assert '[hidden]' in text and '482913' not in text


def test_reading_preview_handles_no_capture_or_live_camera():
    image = reading_preview(None, Snapshot({}, {}, [], []), 'left')
    assert image.shape == (900, 1440, 3)

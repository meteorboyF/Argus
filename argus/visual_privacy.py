"""Local sensitive-text redaction after the mandatory face gate."""
from .sensitive import blur_boxes


def redact_text(face_gated_frame, ocr):
    reading = ocr.read(face_gated_frame)
    return blur_boxes(face_gated_frame,reading.sensitive_boxes())

"""Bounded still selection; callers retain only the best candidate."""
import cv2
import numpy as np


def quality(frame):
    if frame is None or not frame.size:
        return float('-inf')
    gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    usable = float(np.mean((gray > 25) & (gray < 248)))
    return float(np.log1p(sharpness) * usable)

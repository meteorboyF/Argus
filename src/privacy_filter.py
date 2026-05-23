"""Privacy filter — face blurring and EasyOCR text-region detection."""
import cv2
import numpy as np


def blur_faces(image_bgr, face_app, blur_strength=51):
    """Detect faces and apply Gaussian blur — privacy-preserving."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces   = face_app.get(img_rgb)
    result  = image_bgr.copy()
    n_faces = len(faces)
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        pad = 20
        x1  = max(0, x1 - pad)
        y1  = max(0, y1 - pad)
        x2  = min(image_bgr.shape[1], x2 + pad)
        y2  = min(image_bgr.shape[0], y2 + pad)
        roi = result[y1:y2, x1:x2]
        if roi.size > 0:
            blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            result[y1:y2, x1:x2] = blurred
    return result, n_faces


def load_text_detector(model_dir: str, gpu: bool = False):
    """
    Load EasyOCR text detector. Models are auto-downloaded on first use
    (~30 MB) and cached in model_dir.
    """
    import easyocr
    reader = easyocr.Reader(['en'], gpu=gpu, model_storage_directory=model_dir,
                             download_enabled=True, verbose=False)
    return reader


def detect_text_regions(image_bgr, reader, confidence_threshold: float = 0.5):
    """
    Return list of bounding-box quads for detected text regions.
    Each box is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].
    Only returns regions with confidence >= confidence_threshold.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = reader.detect(img_rgb, slope_ths=0.3, height_ths=1.0)
    boxes = []
    if results and results[0]:
        for bbox in results[0]:
            x_min, x_max, y_min, y_max = bbox
            boxes.append([[x_min, y_min], [x_max, y_min],
                          [x_max, y_max], [x_min, y_max]])
    return boxes


def blur_text(image_bgr, reader, blur_strength=31, confidence_threshold=0.5):
    """Detect text regions and blur them — keeps private text out of the World Model."""
    result = image_bgr.copy()
    boxes  = detect_text_regions(image_bgr, reader, confidence_threshold)
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        roi = result[y1:y2, x1:x2]
        if roi.size > 0:
            result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
    return result, len(boxes)

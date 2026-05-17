"""Privacy filter — face blurring and CRAFT text-region detection."""
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


def load_craft(weights_path: str, device: str = "cpu"):
    """
    Load CRAFT text-detection model.

    Requires:
      git clone https://github.com/clovaai/CRAFT-pytorch  (add to PYTHONPATH)
      Download craft_mlt_25k.pth from the CRAFT-pytorch releases.
    """
    import sys, os
    craft_dir = os.path.join(os.path.dirname(__file__), "..", "scripts", "CRAFT-pytorch")
    if craft_dir not in sys.path:
        sys.path.insert(0, craft_dir)

    from craft import CRAFT  # type: ignore
    import torch
    from collections import OrderedDict

    net = CRAFT()
    state = torch.load(weights_path, map_location=device)
    # Strip 'module.' prefix added by DataParallel
    new_state = OrderedDict()
    for k, v in state.items():
        name = k[7:] if k.startswith("module.") else k
        new_state[name] = v
    net.load_state_dict(new_state)
    net = net.to(device).eval()
    return net


def detect_text_regions(image_bgr, craft_net, device: str = "cpu", text_threshold: float = 0.7,
                         link_threshold: float = 0.4, low_text: float = 0.4):
    """Return list of bounding-box polys for detected text regions."""
    import sys, os
    craft_dir = os.path.join(os.path.dirname(__file__), "..", "scripts", "CRAFT-pytorch")
    if craft_dir not in sys.path:
        sys.path.insert(0, craft_dir)

    import torch
    import craft_utils  # type: ignore
    import imgproc      # type: ignore

    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image_bgr, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        y, _ = craft_net(x)

    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    boxes, _ = craft_utils.getDetBoxes(score_text, score_link, text_threshold,
                                        link_threshold, low_text, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return boxes

"""World Model Builder — fuses detector, segmentor, and face detector into a structured scene dict."""
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision.transforms as T

ARGUS_CLASSES = {
    0: 'background', 1: 'navigable_floor', 2: 'wall',
    3: 'door_closed',  4: 'door_open',       5: 'stairs_up',
    6: 'stairs_down',  7: 'person',           8: 'furniture', 9: 'clutter'
}

SEG_TRANSFORM = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WorldModelBuilder:
    def __init__(self, detector, segmentor, face_app, device):
        self.detector  = detector
        self.segmentor = segmentor
        self.face_app  = face_app
        self.device    = device

    def process_frame(self, wide_frame_bgr, depth_map=None):
        """Process one frame and return updated World Model dict."""
        h, w = wide_frame_bgr.shape[:2]
        world = {
            "timestamp"            : time.time(),
            "objects"              : [],
            "navigable_floor"      : True,
            "nearest_obstacle_dist": 999.0,
            "hazard"               : None,
            "segmentation_summary" : "",
        }

        # ── Object Detection ──────────────────────────────
        results = self.detector(wide_frame_bgr, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls)
            label  = results.names[cls_id]
            conf   = float(box.conf)
            if conf < 0.35:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / w
            direction = ('left' if cx < 0.35 else 'right' if cx > 0.65 else 'center')

            if depth_map is not None:
                cy    = int((y1 + y2) / 2)
                cx_px = int((x1 + x2) / 2)
                dist  = float(np.median(
                    depth_map[max(0, cy-10):cy+10, max(0, cx_px-10):cx_px+10]
                ))
            else:
                box_h = (y2 - y1) / h
                dist  = max(0.3, 1.0 / (box_h + 0.01))

            world["objects"].append({
                "label"    : label,
                "distance" : round(dist, 2),
                "direction": direction,
                "private"  : False,
                "conf"     : round(conf, 2),
            })

        # ── Privacy: Face Detection ───────────────────────
        if self.face_app is not None:
            faces = self.face_app.get(cv2.cvtColor(wide_frame_bgr, cv2.COLOR_BGR2RGB))
            for face in faces:
                world["objects"].append({
                    "label"    : "person_face",
                    "distance" : 1.5,
                    "direction": "center",
                    "private"  : True,
                })

        # ── Segmentation → Hazard Detection ───────────────
        if self.segmentor is not None:
            pil_img = Image.fromarray(cv2.cvtColor(wide_frame_bgr, cv2.COLOR_BGR2RGB))
            tensor  = SEG_TRANSFORM(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.segmentor(pixel_values=tensor).logits
                logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            seg_mask = logits.argmax(dim=1).squeeze().cpu().numpy()

            has_stairs_down = (seg_mask == 6).sum() > 0.02 * h * w
            has_stairs_up   = (seg_mask == 5).sum() > 0.02 * h * w
            floor_area      = (seg_mask == 1).sum() / (h * w)

            if has_stairs_down:
                world["hazard"] = "stairs_down"
            elif has_stairs_up:
                world["hazard"] = "stairs_up"

            world["navigable_floor"]      = floor_area > 0.1
            world["segmentation_summary"] = (
                f"floor={floor_area:.0%}, hazard={world['hazard'] or 'none'}"
            )

        # ── Nearest obstacle ──────────────────────────────
        if world["objects"]:
            visible = [o for o in world["objects"] if not o.get("private")]
            if visible:
                world["nearest_obstacle_dist"] = min(o["distance"] for o in visible)

        return world

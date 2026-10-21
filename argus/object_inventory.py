"""On-demand essential-object inventory with the existing YOLO11 TensorRT engine.

Direction-only wide-camera observations; no cross-camera distance projection.
"""
import collections
import numpy as np

ESSENTIAL_CLASSES = {
    0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck',13:'bench',
    24:'backpack',25:'umbrella',28:'suitcase',39:'bottle',41:'cup',56:'chair',
    57:'couch',58:'potted plant',59:'bed',60:'dining table',61:'toilet',62:'television',
    63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'phone',68:'microwave',
    69:'oven',71:'sink',72:'refrigerator',73:'book',74:'clock',76:'scissors',
}


def describe(detections, width):
    from .safety import _direction_of
    groups = collections.OrderedDict()
    for det in sorted(detections,key=lambda d:d.score,reverse=True):
        key = det.cls,_direction_of(float((det.box[0]+det.box[2])/2),width)
        groups[key] = groups.get(key,0)+1
    if not groups:
        return 'I did not confidently detect a familiar object in this view. Try another angle or more light.'
    phrases = [f'{count} {name}{"s" if count>1 and name != "person" else ""} {"ahead" if direction=="center" else "to your "+direction}'
               if not (name=='person' and count>1) else f'{count} people {"ahead" if direction=="center" else "to your "+direction}'
               for (name,direction),count in list(groups.items())[:6]]
    return 'In this view: '+', '.join(phrases)+'. Distances are not verified.'


def inventory(orch):
    from .movers import decode,letterbox
    from .telemetry import bus
    from .grounding import Detection
    movers = getattr(orch,'_movers',None)
    if not movers:
        return 'The object detector is not ready yet.'
    frame = orch.rig.get_wide_frame()
    if frame is None:
        return 'The camera has no current view.'
    detector = movers[0]
    blob,scale,px,py = letterbox(frame,detector.size)
    raw = next(iter(detector.runner.infer({'images':blob}).values()))[0]
    if orch.tasks.cancelled():
        return ''
    detections = decode(raw,scale,px,py,frame.shape,conf=.4,classes=ESSENTIAL_CLASSES)
    from .safety import _direction_of
    previous_target = orch.memory.target
    for det in detections[:16]:
        orch.memory.observe(det.cls,True,_direction_of(float((det.box[0]+det.box[2])/2),frame.shape[1]))
    orch.memory.target = previous_target if previous_target else (detections[0].cls if len(detections)==1 else '')
    bus.detections([Detection(d.cls,d.score,tuple(map(int,d.box)),tuple(map(int,((d.box[:2]+d.box[2:])/2)))) for d in detections])
    return describe(detections,frame.shape[1])

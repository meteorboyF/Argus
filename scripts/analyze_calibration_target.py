#!/usr/bin/env python3
"""Diagnostic target/model comparison on held-out corner observations.

Fits on all but every fifth view. No pruning, quality-gate changes, images,
or calibration activation. Learned grid shape is NOT a measured target.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def heldout_errors(objects, points, indices, k, d):
    errors = []
    for i in indices:
        ok, rv, tv = cv2.solvePnP(objects[i], points[i], k, d)
        if not ok:
            raise RuntimeError('Held-out board pose could not be estimated')
        projected, _ = cv2.projectPoints(objects[i], rv, tv, k, d)
        residual = projected.reshape(-1, 2)-points[i].reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    return errors


def analyze(path):
    with np.load(path, allow_pickle=False) as data:
        objects = list(data['objpoints'])
        points = {'left':list(data['ptsL']), 'right':list(data['ptsR'])}
        size = tuple(map(int, data['image_size']))
        cols = int(data['pattern'][0])
    train = [i for i in range(len(objects)) if i % 5 != 4]
    held = [i for i in range(len(objects)) if i % 5 == 4]
    if len(train) < 8 or len(held) < 2:
        raise ValueError('Need at least 10 views for target analysis')
    report = {'kind':'calibration-target-analysis', 'activated':False,
              'training_views':[i+1 for i in train], 'heldout_views':[i+1 for i in held],
              'cameras':{}, 'cross_camera':{},
              'note':'Held-out poses estimated using corners, intrinsics/shape fit on training views only. '
                     'Learned shape is diagnostic, not physical metrology or approved calibration.'}
    learned, intrinsics = {}, {}
    for name, pts in points.items():
        rows = {}
        for label, flags in (('fixed_k3',cv2.CALIB_FIX_K3), ('free_k3',0),
                             ('rational',cv2.CALIB_RATIONAL_MODEL)):
            rms,k,d,_,_ = cv2.calibrateCamera([objects[i] for i in train],
                                             [pts[i] for i in train],size,None,None,flags=flags)
            rows[label] = {'training_rms_px':float(rms),
                           'heldout_rms_px':heldout_errors(objects,pts,held,k,d)}
            if label == 'fixed_k3':
                intrinsics[name] = k.copy()
        rms,k,d,_,_,shape = cv2.calibrateCameraRO(
            [objects[i] for i in train],[pts[i] for i in train],size,
            cols-1,None,None,flags=cv2.CALIB_FIX_K3)
        shape = shape.reshape(-1,3)
        learned[name] = shape
        rows['released_board'] = {
            'training_rms_px':float(rms),
            'heldout_rms_px':heldout_errors([shape]*len(objects),pts,held,k,d),
            'max_model_coordinate_deviation_mm':
                (np.max(np.abs(shape-objects[0]),axis=0)*1000).tolist()}
        report['cameras'][name] = rows
    # Test each training-derived grid on the other camera's unseen views.
    for source,target in (('left','right'),('right','left')):
        shape,pts = learned[source],points[target]
        rms,k,d,_,_ = cv2.calibrateCamera(
            [shape for _ in train],[pts[i] for i in train],size,intrinsics[target].copy(),None,
            flags=cv2.CALIB_FIX_K3|cv2.CALIB_USE_INTRINSIC_GUESS)
        report['cross_camera'][source+'_shape_on_'+target] = {
            'training_rms_px':float(rms),
            'heldout_rms_px':heldout_errors([shape]*len(objects),pts,held,k,d)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = analyze(args.observations)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

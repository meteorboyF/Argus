#!/usr/bin/env python3
"""Replay numeric calibration observations without cameras or activation.

Example: python3 scripts/diagnose_stereo_observations.py FILE.npz --out REPORT.json
The original views remain unchanged. No pruning or relaxed quality thresholds.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from calibrate_stereo import (calibrate, held_out_row_error, LAST_MONO_RMS,
                              LAST_MONO_FIT, tilt_signature, tilt_spans, quality_gate)


def diagnose(path):
    with np.load(path, allow_pickle=False) as data:
        objects = list(data['objpoints'])
        left, right = list(data['ptsL']), list(data['ptsR'])
        size = tuple(int(v) for v in data['image_size'])
        baseline_mm = float(data['tape_baseline_mm'])
        coverage = len(data['covered'])/9
        pattern = tuple(map(int,data['pattern']))
    rms, kl, dl, kr, dr, rotation, translation = calibrate(objects,left,right,size)
    mono = dict(LAST_MONO_RMS)
    stability = dict(LAST_MONO_FIT)
    spans = tilt_spans([tilt_signature(pts,pattern) for pts in left])
    per_view = []
    for i, obj in enumerate(objects):
        row = {'view':i+1}
        for name, points, k, d in (('left',left[i],kl,dl),('right',right[i],kr,dr)):
            ok, rv, tv = cv2.solvePnP(obj,points,k,d)
            if not ok:
                row[name+'_rms_px'] = None
                continue
            projected, _ = cv2.projectPoints(obj,rv,tv,k,d)
            row[name+'_rms_px'] = float(np.sqrt(np.mean(np.sum(
                (projected.reshape(-1,2)-points.reshape(-1,2))**2,axis=1))))
        per_view.append(row)
    mean, maximum, held = held_out_row_error(objects,left,right,size)
    baseline = float(np.linalg.norm(translation))
    reasons = quality_gate(mono_rms=mono,stereo_rms=rms,row_mean=mean,row_max=maximum,
                           baseline_m=baseline,
                           tape_baseline_m=baseline_mm/1000 if np.isfinite(baseline_mm) else None,
                           n_views=len(objects),coverage=coverage,
                           intrinsic_fit=stability,perspective_spans=spans)
    return {'kind':'stereo-observation-replay','activated':False,
            'views':len(objects),'image_size':size,'mono_rms_px':mono,
            'stereo_rms_px':float(rms),'baseline_m':baseline,
            'intrinsic_fit_stability':stability,'perspective_spans':spans.tolist(),
            'intrinsics_left':kl.tolist(),'intrinsics_right':kr.tolist(),
            'rotation':rotation.tolist(),'translation':translation.tolist(),
            'heldout_views':held,'heldout_mean_px':mean,'heldout_max_px':maximum,
            'coverage':coverage,'rejection_reasons':reasons,'per_view':per_view,
            'note':'Diagnostic fit only. Never writes active calibration. No views removed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = diagnose(args.observations)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Local stationary-board burst diagnostic. SPACE starts; Q/Esc exits.

Private source images stay under /tmp. Report contains only measurements.
No fitting/activation of stereo calibration and no hardware settings changed.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from argus.cameras import CameraRig
from argus.config import load_config
from calibrate_stereo import detect_board, normalize_corners, align_stereo_corners


def analyze(frames, pattern):
    objects = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2).astype(np.float32)
    result = {}
    all_points = {}
    for label in ('classic','sector'):
        corners, errors = [], []
        for frame in frames:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if label == 'classic':
                ok,points = detect_board(gray,pattern)
            else:
                ok,points = cv2.findChessboardCornersSB(gray,pattern,
                                                       flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ok:
                    points = normalize_corners(points,pattern)
            if not ok:
                corners.append(None)
                continue
            if any(p is not None for p in corners):
                reference = next(p for p in corners if p is not None)
                points = align_stereo_corners(reference,points,pattern)
                if points is None:
                    corners.append(None)
                    continue
            corners.append(points)
            h,_ = cv2.findHomography(objects,points,0)
            pred = cv2.perspectiveTransform(objects.reshape(-1,1,2),h)
            errors.append(float(np.sqrt(np.mean(np.sum((pred-points)**2,axis=2)))))
        valid = [p for p in corners if p is not None]
        raw_motion,local_residuals = [], []
        if valid:
            reference = valid[0]
            for points in valid[1:]:
                raw_motion.append(float(np.sqrt(np.mean(np.sum((points-reference)**2,axis=2)))))
                h,_ = cv2.findHomography(points,reference,0)
                aligned = cv2.perspectiveTransform(points,h)
                local_residuals.append(float(np.sqrt(np.mean(np.sum((aligned-reference)**2,axis=2)))))
        result[label] = {
            'detected':len(valid),'attempts':len(frames),
            'planar_grid_rms_px':errors,
            'motion_from_first_rms_px':raw_motion,
            'local_repeatability_after_projective_alignment_px':local_residuals}
        all_points[label] = corners
    differences = []
    for classic,sector in zip(all_points['classic'],all_points['sector']):
        if classic is None or sector is None:
            continue
        sector = align_stereo_corners(classic,sector,pattern)
        if sector is not None:
            differences.append(float(np.sqrt(np.mean(np.sum((classic-sector)**2,axis=2)))))
    result['detector_disagreement_rms_px'] = differences
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    cv2.setNumThreads(1)
    rig = CameraRig(load_config().camera)
    title = 'ARGUS stationary board - SPACE starts 5s burst - Q exits'
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,1600,600)
    private_dir = Path(tempfile.mkdtemp(prefix='argus-stationary-board-'))
    phase = 'preview'
    frames = {'left':[],'right':[]}
    skews = []
    message = 'Bright even light. Support board and rig. SPACE starts; hold still for 5 seconds.'
    deadline = next_sample = 0
    worker = None

    def process():
        nonlocal phase,message
        try:
            report = {'kind':'stationary-board-diagnostic','activated':False,
                      'pair_arrival_skew_ms':skews,'private_images':str(private_dir),
                      'cameras':{},'note':'Planar-grid error also includes lens distortion and target '
                      'geometry. Projective alignment removes global movement, not a guarantee '
                      'of stationary board. No calibration acceptance from this test.'}
            for name,images in frames.items():
                for i,frame in enumerate(images):
                    cv2.imwrite(str(private_dir/f'{name}-{i:02}.png'),frame)
                report['cameras'][name] = analyze(images,(7,7))
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(report,indent=2)+'\n')
            message = 'DONE - measurements saved. Keep preview open or Q to close.'
            phase = 'done'
            print('Report: '+str(args.out),flush=True)
            print('Private images: '+str(private_dir),flush=True)
        except Exception as exc:
            message = 'FAILED: '+type(exc).__name__+' - see terminal'
            phase = 'done'
            print(str(exc),flush=True)

    try:
        while True:
            now = time.monotonic()
            panes=[]
            for name in ('left','right'):
                frame,ts,_ = getattr(rig,name).latest()
                fresh = frame is not None and 0<=time.perf_counter()-ts<=.5
                pane = frame.copy() if fresh else np.zeros((600,960,3),np.uint8)
                cv2.rectangle(pane,(0,0),(pane.shape[1],45),(0,0,0),-1)
                cv2.putText(pane,name.upper()+(' LIVE' if fresh else ' WAITING'),(12,31),0,.8,(80,255,80),2)
                panes.append(pane)
            if phase == 'capture' and now >= next_sample and len(skews)<16:
                pair = rig.get_stereo_pair()
                if pair is not None:
                    frames['left'].append(pair.left.copy())
                    frames['right'].append(pair.right.copy())
                    skews.append(pair.skew_ms)
                    next_sample=now+.3
            if phase == 'capture':
                message=f'HOLD STILL - {len(skews)}/16 synchronized pairs'
                if len(skews)>=16 or now>=deadline:
                    if len(skews)<8:
                        message='Too few pairs. Check USB links; Q exits.'
                        phase='done'
                    else:
                        phase='analysis'; message='Analyzing saved burst - you can put the board down.'
                        worker=threading.Thread(target=process)
                        worker.start()
            canvas=np.vstack([np.hstack(panes),np.zeros((65,1920,3),np.uint8)])
            cv2.putText(canvas,message,(12,640),0,.7,(0,255,255),2)
            cv2.imshow(title,canvas)
            key=cv2.waitKey(25)&255
            if key in (27,ord('q')):
                break
            try:
                if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                    break
            except cv2.error:
                break
            if key==32 and phase=='preview':
                phase='capture'; deadline=now+8; next_sample=now
        if worker is not None:
            worker.join()
    finally:
        rig.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

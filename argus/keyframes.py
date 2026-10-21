"""Bounded sparse keyframe map with geometric ORB/PnP relocalisation.

No place recognition is trusted without descriptor uniqueness, RANSAC inliers,
positive depth and reprojection checks. This is local sparse mapping, not an
optimised global pose graph. No frame image is retained or saved.
"""
from __future__ import annotations
import collections
import cv2
import numpy as np


class KeyframeMap:
    def __init__(self, K, limit=24, min_inliers=15):
        self.K, self.min_inliers = K, min_inliers
        self.frames = collections.deque(maxlen=limit)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def add(self, xyz, descriptors, pose, timestamp):
        if descriptors is None or xyz is None or len(xyz) < self.min_inliers:
            return False
        if self.frames:
            previous = self.frames[-1]
            relative = np.linalg.inv(previous['pose']) @ pose
            angle = np.degrees(np.arccos(np.clip((np.trace(relative[:3,:3])-1)/2,-1,1)))
            if np.linalg.norm(relative[:3,3]) < .3 and angle < 12:
                return False
        world = xyz @ pose[:3,:3].T + pose[:3,3]
        self.frames.append({'xyz':world.astype(np.float64), 'descriptors':descriptors.copy(),
                            'pose':pose.copy(), 'timestamp':timestamp})
        return True

    def relocalize(self, points, descriptors, timestamp, minimum_age_s=0):
        if descriptors is None or len(descriptors) < self.min_inliers:
            return None
        best = None
        for frame in reversed(self.frames):
            if timestamp-frame['timestamp'] < minimum_age_s:
                continue
            matches = self.matcher.knnMatch(frame['descriptors'],descriptors,k=2)
            good = [pair[0] for pair in matches if len(pair)==2 and pair[0].distance < .7*pair[1].distance and pair[0].distance <= 55]
            unique = {}
            for m in sorted(good,key=lambda m:m.distance):
                unique.setdefault(m.trainIdx,m)
            good = list(unique.values())
            if len(good) < self.min_inliers:
                continue
            xyz = frame['xyz'][[m.queryIdx for m in good]]
            uv = np.asarray(points)[[m.trainIdx for m in good]].reshape(-1,2).astype(np.float64)
            ok,rvec,tvec,inliers = cv2.solvePnPRansac(xyz,uv,self.K,None,iterationsCount=150,
                        reprojectionError=2,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or inliers is None or len(inliers) < self.min_inliers or len(inliers)/len(good) < .6:
                continue
            indices = inliers.reshape(-1)
            projected,_ = cv2.projectPoints(xyz[indices],rvec,tvec,self.K,None)
            error = np.linalg.norm(projected.reshape(-1,2)-uv[indices],axis=1)
            R,_ = cv2.Rodrigues(rvec)
            camera = xyz[indices] @ R.T+tvec.reshape(3)
            if np.median(error) > 1.5 or np.mean(camera[:,2] > .1) < .95:
                continue
            world_to_camera = np.eye(4)
            world_to_camera[:3,:3],world_to_camera[:3,3] = R,tvec.reshape(3)
            pose = np.linalg.inv(world_to_camera)
            if not np.isfinite(pose).all():
                continue
            result = (pose,len(indices),len(good))
            if best is None or result[1] > best[1]:
                best = result
        return best

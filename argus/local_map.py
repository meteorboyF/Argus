"""Bounded session occupancy map and obstacle-inflated A* over observed cells.

Unknown/expired cells are impassable. A route is a geometric plan, not evidence
of safe footing, accessible doors or a verified walk. No images persist.
"""
from __future__ import annotations
import heapq
import math
import threading
import time
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Place:
    name: str
    x: float
    z: float
    epoch: int


class LocalMap:
    def __init__(self, size=96, resolution=.25, radius=.4, ttl_s=30, clock=time.monotonic):
        self.size, self.resolution, self.radius, self.ttl_s = size, resolution, radius, ttl_s
        self.clock = clock
        self.evidence = np.zeros((size,size), np.int8)
        self.seen = np.full((size,size), -np.inf)
        self.epoch = None
        self.places = {}
        self._lock = threading.RLock()

    def cell(self, x, z):
        return int(math.floor(x/self.resolution))+self.size//2, int(math.floor(z/self.resolution))+self.size//2

    def world(self, cell):
        return tuple((v-self.size//2+.5)*self.resolution for v in cell)

    def inside(self, cell):
        return all(0 <= v < self.size for v in cell)

    def reset(self, epoch):
        with self._lock:
            self.epoch = epoch
            self.evidence.fill(0)
            self.seen.fill(-np.inf)
            self.places.clear()

    def integrate(self, depth, pose, fx, cx, source_ts):
        now = self.clock()
        if (not pose.tracking or not pose.pose_valid or pose.quality < .5
                or abs(pose.timestamp-source_ts) > .15 or not 0 <= time.perf_counter()-source_ts <= .5):
            return False
        if not all(math.isfinite(v) for v in (pose.x_m,pose.z_m,pose.yaw_deg,fx,cx)) or fx <= 0:
            return False
        h,w = depth.shape[:2]
        yaw = math.radians(pose.yaw_deg)
        origin = self.cell(pose.x_m, pose.z_m)
        if not self.inside(origin):
            return False
        with self._lock:
            if self.epoch != pose.epoch:
                self.reset(pose.epoch)
            free, occupied = set(), set()
            for u in range(0, w, max(1,w//48)):
                values = depth[int(h*.35):int(h*.7), u]
                values = values[np.isfinite(values) & (values > .1)]
                if values.size < max(3, int(h*.15)):
                    continue
                z = float(np.percentile(values,10))
                if z > 6:
                    z = 6
                    endpoint_obstacle = False
                else:
                    endpoint_obstacle = True
                x = (u-cx)*z/fx
                wx = pose.x_m + math.cos(yaw)*x + math.sin(yaw)*z
                wz = pose.z_m - math.sin(yaw)*x + math.cos(yaw)*z
                end = self.cell(wx,wz)
                if not self.inside(end):
                    continue
                steps = max(abs(end[0]-origin[0]),abs(end[1]-origin[1]),1)
                cells = [(int(round(origin[0]+(end[0]-origin[0])*t/steps)),
                          int(round(origin[1]+(end[1]-origin[1])*t/steps))) for t in range(steps+1)]
                free.update(cells[:-1] if endpoint_obstacle else cells)
                if endpoint_obstacle:
                    occupied.add(end)
            for cell in free-occupied:
                self.evidence[cell] = max(-6,int(self.evidence[cell])-1)
                self.seen[cell] = now
            for cell in occupied:
                self.evidence[cell] = min(6,int(self.evidence[cell])+3)
                self.seen[cell] = now
        return True

    def mark(self, name, pose):
        if (not pose.tracking or not pose.pose_valid or pose.quality < .5
                or not all(math.isfinite(v) for v in (pose.x_m,pose.z_m))):
            return False
        with self._lock:
            if self.epoch is None:
                self.reset(pose.epoch)
            if pose.epoch != self.epoch or len(self.places) >= 32 and name not in self.places:
                return False
            self.places[name] = Place(name,pose.x_m,pose.z_m,pose.epoch)
        return True

    def route(self, x, z, name):
        with self._lock:
            place = self.places.get(name)
            if place is None or place.epoch != self.epoch:
                return []
            start, goal = self.cell(x,z), self.cell(place.x,place.z)
            free = (self.evidence < 0) & (self.clock()-self.seen <= self.ttl_s)
            # Inflate every unknown or obstacle cell by the wearer's radius.
            import cv2
            r = int(math.ceil(self.radius/self.resolution))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
            blocked = cv2.dilate((~free).astype(np.uint8),kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
            if not self.inside(start) or not self.inside(goal) or blocked[start] or blocked[goal]:
                return []
            frontier = [(0,start)]
            cost, previous = {start:0}, {}
            while frontier:
                _, current = heapq.heappop(frontier)
                if current == goal:
                    path = [current]
                    while current != start:
                        current = previous[current]
                        path.append(current)
                    return [self.world(c) for c in reversed(path)]
                for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                    nxt = current[0]+dx,current[1]+dz
                    if not self.inside(nxt) or blocked[nxt]:
                        continue
                    new = cost[current]+1
                    if new < cost.get(nxt,math.inf):
                        cost[nxt], previous[nxt] = new,current
                        heapq.heappush(frontier,(new+abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1]),nxt))
        return []


class MappingWorker:
    """Latest snapshots only; ray integration never runs on the safety thread."""
    def __init__(self, orch):
        self.orch, self.map = orch, LocalMap()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run,name='argus-map',daemon=True)
        self._thread.start()

    def _run(self):
        last_ts = None
        while not self._stop.wait(.5):
            o = self.orch
            if o.slam is None or not o.depth.calibrated:
                continue
            health = getattr(getattr(o.depth,'health',None),'state',None)
            if getattr(health,'value',None) != 'ok':
                if self.map.epoch is not None:
                    self.map.reset(None)
                continue
            snapshot = o.latest_observation()
            if snapshot is None:
                continue
            _,depth,ts = snapshot
            if ts == last_ts or depth is None:
                continue
            try:
                intr = o.depth.intrinsics()
                left = snapshot[0]
                if depth.shape[:2] != left.shape[:2]:
                    import cv2
                    depth = cv2.resize(depth,(left.shape[1],left.shape[0]),interpolation=cv2.INTER_NEAREST)
                self.map.integrate(depth,o.slam.state,intr['fx'],intr['cx'],ts)
                last_ts = ts
            except Exception:
                # An optional map failure cannot affect the safety heartbeat.
                from .telemetry import bus
                bus.set(mapping='unavailable')

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

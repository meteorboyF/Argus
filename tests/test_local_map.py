"""Planning blocks unknown, stale and insufficiently wide space."""
from types import SimpleNamespace
import time
import numpy as np

from argus.local_map import LocalMap
from argus.keyframes import KeyframeMap
from argus.assistance import run


def pose(**kwargs):
    return SimpleNamespace(**dict(tracking=True,pose_valid=True,quality=.9,x_m=0.,z_m=0.,yaw_deg=0.,
                                  epoch=0,timestamp=time.perf_counter(),**kwargs))


def test_unknown_and_expired_cells_have_no_route():
    clock=[0]
    m=LocalMap(size=32,clock=lambda:clock[0])
    assert m.mark('door',pose())
    assert m.route(0,1,'door') == []
    m.evidence.fill(-2)
    m.seen.fill(0)
    assert m.route(0,1,'door')
    clock[0]=31
    assert m.route(0,1,'door') == []


def test_obstacle_inflation_prevents_narrow_route():
    m=LocalMap(size=32)
    m.mark('door',pose())
    m.evidence.fill(-2)
    m.seen.fill(m.clock())
    m.evidence[:,18]=4  # wall across the z direction
    assert not m.route(0,2,'door')
    m.evidence[14:19,18]=-2  # 1.25 m opening accommodates 0.8 m inflated body
    assert m.route(0,2,'door')
    m.evidence[16,18]=4
    assert not m.route(0,2,'door')


def test_unmatched_or_invalid_pose_never_updates_map():
    m=LocalMap()
    p=pose()
    depth=np.full((60,80),3.)
    assert m.integrate(depth,p,80,40,p.timestamp)
    p.pose_valid=False
    before=m.evidence.copy()
    assert not m.integrate(depth,p,80,40,p.timestamp)
    np.testing.assert_array_equal(before,m.evidence)
    p.pose_valid=True
    p.timestamp-=1
    assert not m.integrate(depth,p,80,40,time.perf_counter())


def test_new_epoch_forgets_old_places_and_obstacles():
    m=LocalMap()
    m.mark('home',pose())
    m.reset(1)
    assert not m.places and not m.evidence.any()


def test_route_speech_stays_disabled_without_acceptance():
    m=LocalMap(size=32)
    p=pose()
    m.mark('desk',p)
    m.evidence.fill(-1)
    m.seen.fill(m.clock())
    o=SimpleNamespace(depth=SimpleNamespace(calibrated=True,health=SimpleNamespace(state=SimpleNamespace(value='ok'))),slam=SimpleNamespace(state=p),
                      mapping=SimpleNamespace(map=m),cfg=SimpleNamespace(navigation=SimpleNamespace(guided_routes_enabled=False)))
    assert 'disabled' in run(o,'navigate','desk')
    o.depth.calibrated=False
    assert 'unavailable' in run(o,'navigate','desk')


def test_keyframe_relocalization_recovers_pose_from_geometry():
    import cv2
    rng=np.random.default_rng(4)
    xyz=rng.uniform([-1,-.7,2],[1,.7,5],size=(80,3))
    descriptors=rng.integers(0,256,size=(80,32),dtype=np.uint8)
    K=np.array([[300.,0,160],[0,300.,120],[0,0,1]])
    mapping=KeyframeMap(K,min_inliers=15)
    assert mapping.add(xyz,descriptors,np.eye(4),0)
    tvec=np.array([-.2,0,0.])
    points,_=cv2.projectPoints(xyz,np.zeros(3),tvec,K,None)
    found=mapping.relocalize(points,descriptors,1)
    assert found is not None
    np.testing.assert_allclose(found[0][:3,3],[.2,0,0],atol=.01)
    assert found[1] >= 70
    noise=rng.uniform([0,0],[320,240],size=(80,2))
    assert mapping.relocalize(noise,descriptors,1) is None


def test_keyframes_are_bounded():
    mapping=KeyframeMap(np.eye(3),limit=2,min_inliers=3)
    for i in range(4):
        p=np.eye(4)
        p[0,3]=i
        mapping.add(np.ones((3,3)),np.ones((3,32),np.uint8),p,i)
    assert len(mapping.frames)==2

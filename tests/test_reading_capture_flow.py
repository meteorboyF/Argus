from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from argus.reading import page_motion,ReadingSession
from argus.ocr import Reading,Word,order_lines


def test_sensor_grain_does_not_look_like_hand_motion():
    rng=np.random.default_rng(4)
    a=np.clip(120+rng.normal(0,9,(160,200)),0,255).astype(np.uint8)
    b=np.clip(120+rng.normal(0,9,(160,200)),0,255).astype(np.uint8)
    assert np.abs(a.astype(float)-b).mean()>6
    assert page_motion(b,a,(20,20,180,140))<6


def test_background_changes_do_not_block_a_still_page():
    a=np.full((160,200),100,np.uint8)
    b=a.copy(); b[:30]=255
    assert page_motion(b,a,(50,60,150,130))==0
    b[60:130,50:150]=180
    assert page_motion(b,a,(50,60,150,130))>6


def test_coach_measures_short_pairs_instead_of_between_detector_calls():
    rng=np.random.default_rng(7)
    image=rng.integers(60,140,(600,960,3),dtype=np.uint8)
    later=image+25
    frames=iter([image,image,later,later])
    rig=SimpleNamespace(get_reading_frame=lambda:next(frames))
    speaker=SimpleNamespace(speak=Mock(),busy=False)
    box=np.array([[200,150],[750,150],[750,450],[200,450]],np.float32)
    session=ReadingSession(rig,speaker,ocr=SimpleNamespace())
    session._detect=lambda _: [box]
    assert session.coach()
    np.testing.assert_array_equal(session._coached_frame,later)
    assert session.last_check.motion==0


def test_stereo_ocr_uses_the_approved_frame(monkeypatch):
    monkeypatch.setattr('argus.reading.earcons.play',lambda *_:None)
    approved=np.full((30,40,3),100,np.uint8)
    def unexpected():
        raise AssertionError('Must not replace approved stereo view with a new frame')
    rig=SimpleNamespace(cfg=SimpleNamespace(reading_source='left'),capture_reading_still=unexpected)
    speaker=SimpleNamespace(speak=Mock(),cancel_normal=Mock())
    box=np.array([[0,0],[20,0],[20,10],[0,10]],np.float32)
    reading=Reading(order_lines([Word('Test page',box,.9)]),.1,False)
    def recognize(frame,*_):
        np.testing.assert_array_equal(frame,approved)
        return reading
    session=ReadingSession(rig,speaker,ocr=SimpleNamespace(read_robust=recognize))
    def coach():
        session._coached_frame=approved.copy()
        return True
    session.coach=coach
    assert session.read(intro=None,play=False) is reading
    speaker.speak.assert_called_once_with('Captured. Reading now.')

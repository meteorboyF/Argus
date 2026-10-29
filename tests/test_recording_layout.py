import time

import numpy as np

from argus.recording_layout import RecordingLayout,assistant_state,text_tile,wrap,font
from argus.telemetry import Event,Snapshot


def test_long_text_wraps_and_truncates_inside_column():
    face=font(29)
    lines=wrap('A'*300+' '+'a sentence '*200,face,510,5)
    assert len(lines)==5 and lines[-1].endswith('…')
    assert all(face.getlength(line)<=510 for line in lines)


def test_ocr_and_transcript_sensitive_text_masked_before_drawing(monkeypatch):
    from PIL import ImageDraw
    seen=[]
    original=ImageDraw.ImageDraw.text
    def draw_text(self,xy,text,*args,**kwargs):
        seen.append(str(text));return original(self,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',draw_text)
    text_tile.cache_clear()
    snap=Snapshot({},dict(reading_ocr_text='Your OTP is 482913.'),
                  [Event(time.monotonic(),'user','My OTP is 482913.')],[])
    output=RecordingLayout('reading').render(snap,None)
    assert output.shape==(1080,1920,3)
    assert '482913' not in '\n'.join(seen) and '[hidden]' in '\n'.join(seen)


def test_safety_banner_remains_above_diagnostics():
    snap=Snapshot({},dict(calibrated=False),[Event(time.monotonic(),'safety','Stop.')],[])
    output=RecordingLayout('seat').render(snap,None,diagnostics=True)
    np.testing.assert_array_equal(output[730,1100],[60,48,175])


def test_layout_does_not_mutate_camera_frame_and_handles_missing_frame():
    camera=np.full((600,960,3),127,np.uint8)
    before=camera.copy()
    layout=RecordingLayout('conversation')
    snap=Snapshot({},dict(speaking='Hello.'),[],[])
    assert layout.render(snap,camera,recording=True,elapsed=75).shape==(1080,1920,3)
    assert layout.render(snap,None).shape==(1080,1920,3)
    np.testing.assert_array_equal(camera,before)


def test_states_reflect_ocr_and_voice_work():
    assert assistant_state({'speaking':'Hello'})=='Speaking'
    assert assistant_state({'task_busy':True})=='Processing'
    assert assistant_state({'reading_ocr_status':'Framing page','task_busy':False})=='Listening'
    assert assistant_state({'gemma':'transcribing'})=='Understanding'
    assert assistant_state({})=='Listening'

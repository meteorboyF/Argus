"""Synthetic safety and lifecycle checks for the deferred-acceptance build."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from argus.acceptance import cer, score
from argus.ocr import OCREngine, Word, order_lines, speakable_chunks
from argus.service import healthy, stop_child
from argus.still_quality import quality
from argus.trt_runner import TRTRunner


def word(text, x, y, width=80):
    return Word(text, np.array([[x,y],[x+width,y],[x+width,y+20],[x,y+20]], dtype=np.float32), .9)


def test_columns_finish_left_before_right():
    words = [word(str(i), x, y) for i, (x,y) in enumerate([(0,0),(300,0),(0,40),(300,40),(0,80),(300,80)])]
    assert [l.text for l in order_lines(words)] == ['0','2','4','1','3','5']


def test_long_speech_chunks_are_bounded_without_losing_words():
    text = ' '.join(['sentence']*100)+'.'
    chunks = speakable_chunks(order_lines([word(text, 0, 0)]), max_chars=80)
    assert max(map(len, chunks)) <= 80
    assert ' '.join(chunks) == text


def test_upside_down_retry_maps_boxes_back():
    engine = OCREngine.__new__(OCREngine)
    engine.min_score = .5
    box = np.array([[2,3],[12,3],[12,8],[2,8]])
    engine._ocr = Mock(side_effect=[(None,None), ([(box,'Readable',.9)],None)])
    reading = engine.read(np.zeros((20,30,3), np.uint8))
    assert reading.text == 'Readable'
    np.testing.assert_array_equal(reading.lines[0].words[0].box, [29,19]-box)


def test_blank_still_scores_below_textured():
    flat = np.full((240,320,3), 128, np.uint8)
    textured = flat.copy()
    textured[::4] = 50
    assert quality(textured) > quality(flat)
    assert quality(None) == float('-inf')


def test_trt_calls_serialize_and_return_owned_output():
    runner = TRTRunner.__new__(TRTRunner)
    runner._infer_lock = threading.Lock()
    runner._ctx = SimpleNamespace(push=Mock(), pop=Mock())
    buffer = np.zeros(1)
    active = []
    def infer(feeds):
        assert not active
        active.append(True)
        buffer[0] = feeds['x']
        time.sleep(.01)
        active.pop()
        return {'out':buffer}
    runner._infer = infer
    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(runner.infer({'x':i})['out'])) for i in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert sorted(float(a[0]) for a in results) == [0,1,2,3]
    assert runner._ctx.push.call_count == runner._ctx.pop.call_count == 4


def test_cer_unicode_and_empty_reference():
    assert cer('abc', 'axc') == pytest.approx(1/3)
    assert cer('é', 'e\u0301') == 0
    with pytest.raises(ValueError): cer('', '')


@pytest.mark.parametrize('kind', ['reading','seat','wake','demo'])
def test_absent_acceptance_evidence_cannot_pass(kind):
    assert score(kind, {})['status'] == 'partial'


def test_wake_minimums_and_false_wakes():
    data = {'attempts':[True]*40+[False]*10, 'ambient_hours':2, 'false_wakes':1,
            'sounds':dict.fromkeys(['bangla','english','tv'], True)}
    assert score('wake',data)['status'] == 'passed'
    data['false_wakes'] = 2
    assert score('wake',data)['status'] == 'partial'


def test_service_only_signals_owned_running_child():
    child = Mock()
    child.poll.return_value = 0
    stop_child(child)
    child.send_signal.assert_not_called()
    child.poll.return_value = None
    stop_child(child)
    child.send_signal.assert_called_once()
    assert not healthy('http://127.0.0.1:1')


def test_matched_observation_expires_and_owns_buffers():
    from argus.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o._depth_lock = threading.Lock()
    o._latest_left = np.ones((2,2,3))
    o._latest_depth = np.ones((2,2))
    o._latest_depth_ts = time.perf_counter()
    o.depth = SimpleNamespace(calibrated=True)
    left, depth, stamp = o.latest_observation()
    left[:] = depth[:] = 0
    assert o._latest_left.sum() == 12 and o._latest_depth.sum() == 4
    o._latest_depth_ts -= 1
    assert o.latest_observation() is None


def test_cancel_during_ocr_discards_result(monkeypatch):
    from argus.reading import ReadingSession
    from argus.ocr import Reading
    monkeypatch.setattr('argus.reading.earcons.play', lambda *_: None)
    speaker = SimpleNamespace(cancel_normal=Mock(), speak=Mock())
    rig = SimpleNamespace(capture_reading_still=lambda: np.ones((20,20,3), np.uint8))
    ocr = SimpleNamespace()
    session = ReadingSession(rig, speaker, ocr=ocr)
    def recognize(*_):
        session.stop()
        return Reading(order_lines([word('Never speak this result',0,0)]), .1, False)
    ocr.read_robust = recognize
    assert session.read(coach=False, intro=None) is None
    assert session.last is None
    speaker.speak.assert_not_called()
    from argus.telemetry import bus
    assert not bus.get('reading_ocr_text')


def test_bangla_requires_installed_language(monkeypatch):
    from argus.bangla_ocr import BanglaOCR
    monkeypatch.setattr('argus.bangla_ocr.subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='List of languages:\neng\n'))
    with pytest.raises(RuntimeError, match='ben'):
        BanglaOCR()


def test_still_reopens_preview_when_open_raises(monkeypatch):
    import argus.cameras as cameras
    from argus.config import CameraConfig
    rig = cameras.CameraRig.__new__(cameras.CameraRig)
    rig.cfg = CameraConfig()
    rig.wide = SimpleNamespace(release=Mock())
    rig.wide_index = 2
    restored = object()
    monkeypatch.setattr(cameras, '_open', Mock(side_effect=RuntimeError('USB gone')))
    reader = Mock(return_value=restored)
    monkeypatch.setattr(cameras, '_Reader', reader)
    with pytest.raises(RuntimeError, match='USB gone'):
        rig.capture_still()
    assert rig.wide is restored


def test_seat_exception_removes_whitelist(monkeypatch):
    from argus.seat import SeatSession
    monkeypatch.setattr('argus.movers.letterbox', lambda f: (f, 1, 0, 0))
    det = SimpleNamespace(runner=SimpleNamespace(infer=Mock(side_effect=RuntimeError('inference failed'))))
    o = SimpleNamespace(_movers=(det,None,None), depth=SimpleNamespace(calibrated=False),
                        cfg=SimpleNamespace(depth=SimpleNamespace(focal_px=500)),
                        speaker=SimpleNamespace(speak=Mock()), seat_cancel=threading.Event(),
                        _stop=threading.Event(), corridor_whitelist=(1, time.monotonic()+1),
                        latest_observation=lambda: (np.zeros((2,2,3)),None,time.perf_counter()))
    with pytest.raises(RuntimeError): SeatSession(o).run()
    assert o.corridor_whitelist is None


def test_rotated_page_keeps_recognition_order_and_source_boxes():
    engine = OCREngine.__new__(OCREngine)
    engine.min_score = .5
    top, bottom = word('First', 0, 0), word('Second', 0, 40)
    engine._ocr = Mock(side_effect=[(None,None), ([(top.box,top.text,.9),(bottom.box,bottom.text,.9)],None)])
    reading = engine.read(np.zeros((100,100,3), np.uint8))
    assert reading.text == 'First\nSecond'
    assert reading.lines[0].y0 > reading.lines[1].y0


def test_supervisor_leaves_external_server_running(monkeypatch, tmp_path):
    import argus.service as service
    child = Mock()
    child.poll.return_value = 0
    cfg = SimpleNamespace(agent=SimpleNamespace(server_url='http://127.0.0.1:8080'), speech=None)
    monkeypatch.setattr('argus.config.load_config', lambda _: cfg)
    monkeypatch.setattr(service, 'healthy', lambda _: True)
    monkeypatch.setattr(service.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(service.threading, 'Event', lambda: SimpleNamespace(wait=lambda _: False))
    popen = Mock(return_value=child)
    monkeypatch.setattr(service.subprocess, 'Popen', popen)
    assert service.supervise(tmp_path, log_dir=tmp_path/'logs') == 0
    assert popen.call_count == 1  # runtime only
    child.send_signal.assert_not_called()


def test_supervisor_announces_failure_and_returns_nonzero(monkeypatch, tmp_path):
    import argus.service as service
    child = Mock()
    child.poll.return_value = 1
    cfg = SimpleNamespace(agent=SimpleNamespace(server_url='http://127.0.0.1:8080'), speech=None)
    monkeypatch.setattr('argus.config.load_config', lambda _: cfg)
    monkeypatch.setattr(service, 'healthy', lambda _: True)
    monkeypatch.setattr(service.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(service.threading, 'Event', lambda: SimpleNamespace(wait=lambda _: False))
    monkeypatch.setattr(service.subprocess, 'Popen', Mock(return_value=child))
    guide = Mock()
    context = Mock(__enter__=Mock(return_value=guide), __exit__=Mock(return_value=False))
    monkeypatch.setattr('argus.guide.Guide', Mock(return_value=context))
    assert service.supervise(tmp_path, log_dir=tmp_path/'logs') == 1
    guide.failed.assert_called_once()

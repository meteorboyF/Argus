from pathlib import Path
from types import SimpleNamespace

from argus.demo_view import DemoView


def test_hidden_reset_preserves_old_take_and_skips_existing_files(tmp_path, monkeypatch):
    import argus.demo_view as module
    saved = []

    class FakeRecorder:
        def __init__(self, path, fps):
            self.out, self.fps = Path(path), fps
        def finish(self):
            saved.append(self.out)
            return str(self.out)

    monkeypatch.setattr(module, 'Recorder', FakeRecorder)
    old = tmp_path / 'scene-take-03.mp4'
    (tmp_path / 'scene-take-04.mp4').touch()
    (tmp_path / 'scene-take-05.jsonl').touch()
    view = DemoView.__new__(DemoView)
    view.recorder = FakeRecorder(old, 15)
    view.typed = 'unfinished request'
    view._reset_at = -10
    view._recording_finalizers = []
    assert view._handle_key(ord('R'))
    for worker in view._recording_finalizers:
        worker.join()
    assert saved == [old]
    assert view.recorder.out.name == 'scene-take-06.mp4'
    assert view.typed == ''
    # Held key/repeated event cannot create another clip immediately.
    view._handle_key(ord('r'))
    assert view.recorder.out.name == 'scene-take-06.mp4'


def test_r_without_recording_remains_available_for_typing():
    view = DemoView.__new__(DemoView)
    view.recorder = None
    view.typed = 'whe'
    view.on_question = lambda question: None
    assert view._handle_key(ord('r'))
    assert view.typed == 'wher'


def test_clear_interaction_keeps_safety_and_existing_log():
    from argus.telemetry import Telemetry
    bus = Telemetry()
    logged = []
    bus.sinks.append(logged.append)
    bus.log('user', 'Old request')
    bus.log('argus', 'Old response')
    bus.log('safety', 'Stop')
    bus.set(speaking='Old response', reading_ocr_text='Old page',
            reading_ocr_status='Done', calibrated=False)
    before = list(logged)
    bus.clear_interaction_display()
    snap = bus.snapshot()
    assert [(e.kind, e.text) for e in snap.events] == [('safety', 'Stop')]
    assert not snap.values.get('speaking') and not snap.values.get('reading_ocr_text')
    assert snap.values['calibrated'] is False
    assert logged == before

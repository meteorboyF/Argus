"""The soak's spoken instructions must reach the existing demo window."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def test_guidance_uses_identical_spoken_and_displayed_instruction(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('soak_tool', scripts / 'soak_fast_loop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from argus.telemetry import Telemetry
    telemetry = Telemetry()
    monkeypatch.setattr(module, 'bus', telemetry)
    spoken=[]
    speaker=SimpleNamespace(busy=False, speak=spoken.append,
                            wait_until_idle=lambda **kwargs: True)
    guide=module.DemoGuide(None, speaker=speaker, show=False)
    guide.say('Stay seated. Ask Argus about the room.', force=True)
    guide.show(progress=.5)
    assert telemetry.get('study_instruction') == spoken[-1]
    assert telemetry.get('study_progress') == .5
    guide.failed('The camera stopped.')
    assert telemetry.get('study_instruction') == spoken[-1] == 'Failed. The camera stopped.'

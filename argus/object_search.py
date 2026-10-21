"""Bounded object watch using current wide frames, without metric claims."""
import time
from .grounding import UnsupportedObjectError


def search(orch, name, timeout_s=25, period_s=1):
    from .telemetry import bus
    if not name.strip():
        return 'Which object should I look for?'
    orch.seat_cancel.clear()
    orch.speaker.speak(f'Looking for the {name}. Stay in place and turn slowly if you want to scan.')
    deadline = time.monotonic()+timeout_s
    said_at = time.monotonic()
    hits = 0
    while time.monotonic() < deadline and not orch._stop.is_set() and not orch.seat_cancel.is_set():
        if orch.tasks.cancelled():
            return ''
        frame = orch.rig.get_wide_frame()
        if frame is None:
            hits = 0
            time.sleep(.1)
            continue
        # Grounding is entirely local; no image is sent to a language model.
        det = orch.grounder.find_object(name, frame)
        if orch.tasks.cancelled() or orch.seat_cancel.is_set():
            return ''
        result = orch._fuse_detection(name, det, frame)
        orch.memory.observe(name, det is not None, result.get('direction',''))
        hits = hits+1 if det is not None else 0
        bus.detections([det] if det is not None else [])
        if hits >= 2:
            where = {'left':'to your left','right':'to your right','center':'straight ahead'}.get(result['direction'],'in view')
            return f'I found the {name} {where}. I cannot estimate its distance from this camera.'
        if time.monotonic()-said_at > 6:
            orch.speaker.speak(f'I have not confirmed the {name} yet. Keep it in view if possible.')
            said_at = time.monotonic()
        if orch._stop.wait(period_s):
            break
    return f'I could not confirm the {name} in this scan. Try another angle or better light.'

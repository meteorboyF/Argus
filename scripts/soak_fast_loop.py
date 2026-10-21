#!/usr/bin/env python3
"""Guided, stationary full-stack soak. Human workload confirmation stays pending.

DISPLAY=:0 python3 scripts/soak_fast_loop.py --seconds 600 --out reports/soak-2026-10-21.json
Keep a printed page ready and use the headset. No walking before calibration.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argus.config import load_config
from argus.doctor import oc_events
from argus.guide import Guide
from argus.orchestrator import Orchestrator
from argus.telemetry import bus
from argus.timing import heartbeat_gaps
from profile_fast_loop import memory, stats


class DemoGuide(Guide):
    """Use the existing demo window to avoid a second Qt event loop."""
    def show(self, *args, **kwargs):
        bus.set(study_instruction=self.headline,
                study_progress=kwargs.get('progress', self.progress))
        return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=600)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    import requests
    requests.get(cfg.agent.server_url.rstrip('/')+'/health', timeout=3).raise_for_status()
    o = Orchestrator(cfg)
    guide = DemoGuide(None, speaker=o.speaker, show=False)
    events = collections.deque(maxlen=30000)
    bus.sinks.append(events.append)
    report = dict(kind='argus-full-stack-soak', T1_complete=False,
                  human_workload_confirmation='pending', requested_s=args.seconds,
                  backend=o.depth.backend, calibrated=o.depth.calibrated,
                  config=cfg.source, cv_threads=__import__('cv2').getNumThreads())
    samples=[]
    done=threading.Event()
    def _monitor():
        if not o._fast_ready.wait(30):
            return
        begin=time.monotonic(); oc0=oc_events()
        report.update(begin=begin, oc_before=oc0)
        prompts = [
            'Stay seated. Ask Argus what is in front of you.',
            'Hold up the page. Say Argus, read this. Try stop, repeat, and next page.',
            'Ask Argus a question about the page, then ask about the room.',
        ]
        step=-1
        while not done.is_set() and time.monotonic()-begin < args.seconds:
            elapsed=time.monotonic()-begin
            n=int(elapsed//45)
            if n != step and not o.speaker.busy:
                step=n
                guide.say(prompts[n % len(prompts)], force=True)
            guide.show(progress=min(1, elapsed/args.seconds))
            sample=memory()
            if not samples or int(elapsed)//10 > int(samples[-1]["elapsed_s"])//10:
                try:
                    sample["llama_healthy"] = requests.get(
                        cfg.agent.server_url.rstrip("/")+"/health", timeout=3).status_code == 200
                except requests.RequestException:
                    sample["llama_healthy"] = False
            sample["elapsed_s"]=elapsed
            samples.append(sample)
            done.wait(.5)
        end=time.monotonic()
        captured=list(events)
        ticks=[e for e in captured if e['event']=='fast_tick' and e['t'] >= begin]
        heart=[e['t'] for e in captured if e['event']=='depth_ready']
        gaps=heartbeat_gaps(heart, begin, end)
        timing={k:stats(e.get(k) for e in ticks) for k in
                ('tick_ms','interval_ms','capture_wait_ms','depth_ms','hazards_ms','movers_ms','publish_ms')}
        passed=(end-begin >= args.seconds and not gaps and timing['tick_ms'] is not None
                and timing['tick_ms']['p95'] <= 100
                and max((s['stack_rss_mb'] for s in samples),default=float('inf')) <= 5.5*1024)
        report.update(duration_s=end-begin, timing=timing, outages=gaps,
                      timing_and_rss_pass=passed, resources=samples,
                      llama_healthy_whole_run=all(s.get("llama_healthy", True) for s in samples),
                      reading_attempts=sum(e["event"] in ("reading", "reading_retry") for e in captured),
                      oc_after=oc_events(),
                      heard_count=sum(e['event']=='heard' for e in captured),
                      query_count=sum(e['event']=='query_end' for e in captured))
        out.write_text(json.dumps(report, indent=2))
        # Ending is spoken before runtime teardown; acceptance remains separate.
        if passed:
            guide.done('Timing and resident memory checks passed. Workload review is still required.')
        else:
            guide.failed('Soak acceptance is incomplete. The measurements have been saved.')
        o._exit_requested=True
    def monitor():
        try:
            _monitor()
        except Exception as exc:
            report['monitor_error'] = str(exc)
            report['timing_and_rss_pass'] = False
            out.write_text(json.dumps(report, indent=2))
            guide.failed('The measurement monitor failed. The session will stop.')
        finally:
            o._exit_requested = True
    t=threading.Thread(target=monitor, daemon=True, name='soak-monitor')
    t.start()
    try:
        o.run(demo=True, record=str(out.with_suffix('.mp4')),
              session_log=str(out.with_suffix('.jsonl')))
    finally:
        done.set(); t.join(timeout=35)
        bus.sinks.remove(events.append)
        report['fast_error']=str(o._fast_error) if o._fast_error else None
        report['T1_complete']=False
        out.write_text(json.dumps(report, indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())

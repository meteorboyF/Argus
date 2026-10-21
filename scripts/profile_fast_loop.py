#!/usr/bin/env python3
"""Stationary T1 component isolation. No acceptance claim or saved camera frames.

Run with llama-server up and cameras free. Each phase runs real production
modules; Whisper uses a synthetic spoken query and OCR a synthetic busy page.
This does not replace the human full-stack acceptance soak.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import psutil
from argus.config import load_config
from argus.doctor import oc_events
from argus.orchestrator import Orchestrator
from argus.telemetry import bus, TegraStats
from argus.timing import heartbeat_gaps


def stats(values):
    a = np.asarray([x for x in values if x is not None], dtype=float)
    return None if not len(a) else dict(n=len(a), median=float(np.median(a)),
                                       p95=float(np.percentile(a, 95)), max=float(a.max()))


def memory():
    mem = psutil.virtual_memory()
    proc = psutil.Process()
    children = proc.children(recursive=True)
    servers = [p for p in psutil.process_iter(['name']) if p.info['name'] == 'llama-server']
    rss = 0
    for candidate in {p.pid:p for p in [proc, *children, *servers]}.values():
        try:
            rss += candidate.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    vm = psutil.swap_memory()
    counters = {}
    for line in Path('/proc/vmstat').read_text().splitlines():
        key, value = line.split()
        if key in ('pswpin', 'pswpout', 'pgmajfault'):
            counters[key] = int(value)
    return dict(vmstat=counters, stack_rss_mb=rss / 2**20, available_mb=mem.available / 2**20,
                swap_used_mb=vm.used / 2**20, swap_in=vm.sin, swap_out=vm.sout,
                threads=proc.num_threads(), gr3d=bus.get('gr3d_percent'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--phases', default='baseline,gemma,whisper,faceblur,recorder,ocr,still,combined')
    ap.add_argument('--cv-threads', type=int)
    ap.add_argument('--offscreen', action='store_true', help='render and record without opening a monitor window')
    args = ap.parse_args()
    if args.cv_threads is not None:
        cv2.setNumThreads(args.cv_threads)
    cfg = load_config()
    report = dict(kind='argus-fast-loop-profile', acceptance=False,
                  config=cfg.source, cv_threads=cv2.getNumThreads(), cache_prompt=False, offscreen=args.offscreen,
                  notes=['Stationary, uncalibrated rig; geometry rules requiring calibration are inactive.',
                         'Phases toggle active work; previously loaded model memory remains resident.',
                         'OCR busy page and Whisper speech are synthetic, not human acceptance.'], phases=[])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse an incomplete stack before allocating model/camera resources.
    import requests
    requests.get(cfg.agent.server_url.rstrip('/')+'/health', timeout=3).raise_for_status()
    o = Orchestrator(cfg, enable_audio=False)
    report['cv_threads'] = cv2.getNumThreads()
    report['server_commands'] = [p.cmdline() for p in psutil.process_iter(['name'])
                                 if p.info['name'] == 'llama-server']
    tg = TegraStats(); tg.start()
    events = collections.deque(maxlen=20000)
    last_depth = [None]
    def collect(event):
        events.append(event)
        if event['event'] == 'depth_ready':
            last_depth[0] = event['t']
    bus.sinks.append(collect)
    halt = None; workers = []; view = None
    try:
        o.start_fast_loop()
        time.sleep(2)
        report['backend'] = o.depth.backend
        report['calibrated'] = o.depth.calibrated
        for phase in args.phases.split(','):
            row = dict(phase=phase, operations=[], errors=[])
            halt = threading.Event()
            workers = []
            view = None
            def loop(name, fn, period=None):
                def run():
                    while not halt.is_set():
                        start = time.perf_counter()
                        try:
                            fn()
                            row['operations'].append(dict(name=name, ms=(time.perf_counter()-start)*1000))
                        except Exception as e:
                            row['errors'].append(dict(name=name, error=str(e)))
                        halt.wait(0.1 if period is None else max(0, period-(time.perf_counter()-start)))
                t = threading.Thread(target=run, daemon=True, name='profile-'+name)
                workers.append(t); t.start()
            if phase in ('whisper', 'combined'):
                from argus.speech import Transcriber
                from argus.wake_stt import transcribe_hinted
                from piper import PiperVoice
                v = PiperVoice.load(cfg.speech.piper_voice)
                chunks = list(v.synthesize('Argus, what is in front of me? Please read this page.'))
                audio = np.concatenate([c.audio_float_array for c in chunks])
                sr = chunks[0].sample_rate
                audio = np.interp(np.arange(int(len(audio)*16000/sr))*sr/16000,
                                  np.arange(len(audio)), audio).astype(np.float32)
                stt = Transcriber(cfg.speech)
                loop('whisper', lambda: transcribe_hinted(stt, audio))
            if phase in ('gemma', 'combined'):
                def gemma():
                    frame = o.rig.get_wide_frame()
                    if frame is None: raise RuntimeError('no wide frame')
                    gated, _ = o.privacy.apply(frame)
                    import requests
                    from argus.agent import _encode_image, ARGUS_SYSTEM
                    response = requests.post(o.agent.endpoint, json={
                        'messages': [{'role':'system','content':ARGUS_SYSTEM},
                                     {'role':'user','content':[
                                         {'type':'text','text':'Describe this scene briefly.'},
                                         {'type':'image_url','image_url':{'url':_encode_image(gated, cfg.agent.image_tokens)}}]}],
                        'cache_prompt':False, 'max_tokens':cfg.agent.max_tokens,
                        'temperature':cfg.agent.temperature}, timeout=cfg.agent.request_timeout_s)
                    response.raise_for_status()
                loop('gemma', gemma)
            if phase in ('ocr', 'combined'):
                from argus.ocr import OCREngine
                ocr = OCREngine()
                page = np.full((2400, 1800, 3), 255, np.uint8)
                for y in range(60, 2350, 55):
                    cv2.putText(page, 'A printed page for measuring ARGUS reading latency.',
                                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                loop('ocr', lambda: ocr.read(page))
            if phase in ('faceblur', 'recorder', 'combined'):
                from argus.demo_view import DemoView
                view = DemoView(privacy_cfg=cfg.privacy if phase != 'recorder' else None,
                                fullscreen=False,
                                record_path=str(out.with_suffix('.mp4')) if phase != 'faceblur' else None)
                if args.offscreen:
                    loop('render', lambda: view.render(bus.snapshot()), period=1/15)
                else:
                    view.start()
            if phase == 'still':
                def still():
                    if o.rig.capture_still() is None:
                        raise RuntimeError('12 MP still failed')
                    halt.wait(2)
                loop('still', still)
            # Initialisation is excluded from stage statistics but remains visible
            # to the real watchdog and event stream.
            events.clear()
            begin = time.monotonic(); previous_depth = last_depth[0]
            oc0 = oc_events(); samples=[]
            while time.monotonic()-begin < args.seconds and not o._stop.is_set():
                if view is not None and not args.offscreen and view.closed.is_set():
                    row['errors'].append(dict(name='display', error='demo closed before phase ended'))
                    break
                samples.append(memory()); time.sleep(0.5)
            end = time.monotonic(); halt.set()
            captured = list(events)
            for t in workers:
                t.join(timeout=cfg.agent.request_timeout_s+5)
                if t.is_alive():
                    row['errors'].append(dict(name=t.name, error='worker did not stop'))
            if view:
                view.stop()
                if view.is_alive(): view.join(timeout=5)
                if view.recorder: view.finish()
            ticks = [e for e in captured if e['event']=='fast_tick']
            fields = ('tick_ms','interval_ms','capture_wait_ms','depth_ms','hazards_ms',
                      'movers_ms','publish_ms','navigation_ms','alarms_health_ms','skew_ms')
            row.update(duration_s=end-begin, ticks=len(ticks), valid_depth=sum(e['depth_ok'] for e in ticks),
                       timing={k: stats(e.get(k) for e in ticks) for k in fields},
                       depth_stages={k:stats(e.get('depth_stages',{}).get(k) for e in ticks)
                                     for k in ('rectify_ms','disparity_ms')},
                       depth_gaps=[e for e in captured if e['event']=='depth_gap'],
                       outages=heartbeat_gaps([e['t'] for e in captured if e['event']=='depth_ready'], begin, end, previous_depth),
                       heartbeat_interval_ms=stats(np.diff([e['t'] for e in captured if e['event']=='depth_ready'])*1000),
                       resources=samples, oc_before=oc0, oc_after=oc_events())
            workers = []; view = None
            report['phases'].append(row)
            out.write_text(json.dumps(report, indent=2))
            print(json.dumps({k:row[k] for k in ('phase','ticks','valid_depth','timing','errors')}), flush=True)
            if o._fast_error: raise o._fast_error
    except KeyboardInterrupt:
        report['interrupted'] = True
    finally:
        if halt is not None:
            halt.set()
        for worker in workers:
            worker.join(timeout=cfg.agent.request_timeout_s+5)
        if view is not None:
            view.stop()
            if view.is_alive():
                view.join(timeout=5)
            if view.recorder:
                view.finish()
        bus.sinks.remove(collect)
        o.stop(); tg.stop()
        out.write_text(json.dumps(report, indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare existing vs torch-free CTranslate2 inference in fresh processes.

No audio is played or recorded. Piper generates a fixed synthetic query.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(policy, audio_path, count):
    if policy == 'inference':
        from argus.runtime_imports import enforce_inference_import_budget
        enforce_inference_import_budget()
    import importlib.metadata
    import resource
    import psutil
    import numpy as np
    from argus.config import load_config
    from argus.doctor import oc_events
    from argus.speech import Transcriber
    from argus.wake_stt import transcribe_hinted
    from argus.vad import silero_prob_fn
    from argus.telemetry import bus, TegraStats
    cfg=load_config()
    oc0=oc_events(); tg=TegraStats(); tg.start()
    start=time.perf_counter()
    stt=Transcriber(cfg.speech)
    vad=silero_prob_fn()
    load_s=time.perf_counter()-start
    audio=np.load(audio_path)
    outputs=[]; latency=[]; available=[]; gr3d=[]
    try:
        for _ in range(count):
            start=time.perf_counter()
            outputs.append(transcribe_hinted(stt,audio))
            latency.append((time.perf_counter()-start)*1000)
            available.append(psutil.virtual_memory().available/2**20)
            gr3d.append(bus.get('gr3d_percent'))
        probability=vad(np.zeros(512,np.float32))
    finally:
        tg.stop()
    return dict(policy=policy,backend='Whisper tiny CTranslate2 int8 CPU; Silero ONNX CPU',
                ctranslate2=importlib.metadata.version('ctranslate2'),
                torch_loaded='torch' in sys.modules, load_s=load_s,transcripts=outputs,
                latency_ms=dict(median=float(np.median(latency)),p95=float(np.percentile(latency,95))),
                peak_worker_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                minimum_MemAvailable_mib=min(available),
                peak_gr3d_percent=max((x for x in gr3d if x is not None),default=None),
                vad_silence_probability=probability,
                oc_before=oc0,oc_after=oc_events())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out')
    ap.add_argument('--count',type=int,default=10)
    ap.add_argument('--worker',choices=('existing','inference'))
    ap.add_argument('--audio')
    args=ap.parse_args()
    if args.worker:
        print('RESULT '+json.dumps(worker(args.worker,args.audio,args.count)),flush=True)
        return
    if not args.out:
        ap.error('--out is required')
    import numpy as np
    from piper import PiperVoice
    from argus.config import load_config
    cfg=load_config()
    voice=PiperVoice.load(cfg.speech.piper_voice)
    phrase='Hey Argus, what is in front of me?'
    chunks=list(voice.synthesize(phrase))
    audio=np.concatenate([x.audio_float_array for x in chunks])
    sr=chunks[0].sample_rate
    audio=np.interp(np.arange(int(len(audio)*16000/sr))*sr/16000,
                    np.arange(len(audio)),audio).astype(np.float32)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='argus_stt_memory_') as tmp:
        path=Path(tmp)/'query.npy';np.save(path,audio)
        for policy in ('existing','inference'):
            result=subprocess.run([sys.executable,__file__,'--worker',policy,'--audio',str(path),
                                   '--count',str(args.count)],capture_output=True,text=True,timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr[-2000:])
            row=next(json.loads(line[7:]) for line in result.stdout.splitlines() if line.startswith('RESULT '))
            rows.append(row)
    report=dict(kind='argus-stt-optional-import-memory',synthetic_phrase=phrase,rows=rows,
                transcripts_identical=rows[0]['transcripts']==rows[1]['transcripts'],
                worker_rss_saved_mib=rows[0]['peak_worker_rss_mib']-rows[1]['peak_worker_rss_mib'])
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()

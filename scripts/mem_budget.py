#!/usr/bin/env python3
"""Per-component resident memory of the ARGUS runtime (M3 baseline).

Each component is loaded in a fresh Python subprocess, exercised once, and
measured: process RSS (VmRSS / VmHWM) and the drop in system MemAvailable,
which on Jetson also captures CUDA/NvMap allocations that RSS can miss.

    python3 scripts/mem_budget.py --out reports/mem-budget-2026-10-02.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PRELUDE = r"""
import sys, time, json
sys.path.insert(0, %r)
def avail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'): return int(l.split()[1]) // 1024
def status():
    out = {}
    for l in open('/proc/self/status'):
        if l.startswith(('VmRSS', 'VmHWM')):
            k, v = l.split(':'); out[k] = int(v.split()[0]) // 1024
    return out
a0 = avail(); s0 = status(); t0 = time.perf_counter()
""" % str(REPO)

EPILOGUE = r"""
time.sleep(1.0)
s1 = status()
print('RESULT ' + json.dumps({'base_rss_mb': s0['VmRSS'], 'rss_mb': s1['VmRSS'],
      'hwm_mb': s1['VmHWM'], 'mem_drop_mb': a0 - avail(),
      'load_s': round(time.perf_counter() - t0, 2)}))
"""

COMPONENTS = {
    "python+numpy+cv2": "import numpy, cv2",
    "privacy (InsightFace SCRFD, CPU)": """
from argus.config import PrivacyConfig
from argus.privacy import PrivacyGate
import numpy as np
g = PrivacyGate(PrivacyConfig()); assert g.ready
g.apply(np.zeros((1080, 1920, 3), np.uint8))
""",
    "grounding TRT engine only": """
from argus.config import GroundingConfig
from argus.grounding import Grounder
gr = Grounder(GroundingConfig())
""",
    "grounding TRT + CLIP text (torch CPU)": """
from argus.config import GroundingConfig
from argus.grounding import Grounder
gr = Grounder(GroundingConfig())
gr._embed('chair')
""",
    "grounding TRT + vocab table (warm, no torch)": """
from argus.config import GroundingConfig
from argus.grounding import Grounder
gr = Grounder(GroundingConfig())
gr.warm()
assert not gr.live_encoder_loaded and 'torch' not in sys.modules
""",
    "CUDA SAD stereo": """
import numpy as np
from argus.cuda_stereo import CudaStereoMatcher
m = CudaStereoMatcher()
l = np.random.randint(0, 255, (300, 480), np.uint8)
m.compute(l, np.roll(l, -8, axis=1), 128, 2, 10)
""",
    "Piper TTS": """
from argus.config import SpeechConfig
from piper import PiperVoice
v = PiperVoice.load(SpeechConfig().piper_voice)
list(v.synthesize('ARGUS memory check.'))
""",
    "faster-whisper tiny int8": """
from faster_whisper import WhisperModel
import numpy as np
m = WhisperModel('tiny', device='cpu', compute_type='int8')
list(m.transcribe(np.zeros(16000, np.float32), language='en')[0])
""",
    "openWakeWord": """
from openwakeword.model import Model
import numpy as np
m = Model(); m.predict(np.zeros(1280, np.int16))
""",
}


def measure(name: str, code: str) -> dict:
    src = PRELUDE + code + EPILOGUE
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=600)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r["component"] = name
            return r
    return {"component": name, "error": (p.stderr or p.stdout)[-400:],
            "wall_s": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    rows = []
    for name, code in COMPONENTS.items():
        if args.only and args.only not in name:
            continue
        r = measure(name, code)
        rows.append(r)
        print(json.dumps(r), flush=True)
        time.sleep(2)
    Path(args.out).write_text(json.dumps({"kind": "argus-mem-budget",
                                          "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "rows": rows}, indent=2))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

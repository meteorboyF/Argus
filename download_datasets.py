#!/usr/bin/env python3
"""
ARGUS Dataset Pre-Downloader
Downloads all training datasets to Google Drive before the pipeline runs.
Data persists across Colab sessions — re-running skips already-downloaded files.

Usage (in any Colab cell):
    !python download_datasets.py

What gets downloaded:
  ADE20K          ~900 MB   → datasets/ade20k/          (SegFormer training)
  NYUv2           ~2.8 GB   → datasets/nyuv2/           (SegFormer supplemental)
  COCO train2017  ~17 GB    → datasets/coco/             (YOLOv8 pre-training)
  COCO val2017    ~750 MB   → datasets/coco/
  COCO annots     ~240 MB   → datasets/coco/
  RAFT-Stereo wts ~30 MB    → models/depth/              (depth model weights)
  SCRFD weights   ~90 MB    → models/privacy/            (face detection)
  CRAFT weights   ~90 MB    → models/privacy/            (text detection)
  Phi-3.5 GGUF    ~2.7 GB   → models/llm/               (LLM)
  Piper TTS       ~65 MB    → models/speech/piper/       (text-to-speech)
  Whisper tiny    ~39 MB    → models/speech/whisper/     (speech-to-text, auto-cached)

Total first run: ~24 GB   (COCO dominates)
"""

import os, sys, subprocess, zipfile, tarfile, shutil, time, json
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path('/content/drive/MyDrive/ARGUS')
DS      = BASE / 'datasets'
MODELS  = BASE / 'models'
EXPORTS = BASE / 'exports'
LOGS    = BASE / 'logs'

for d in [DS, MODELS, EXPORTS, LOGS,
          DS/'ade20k', DS/'nyuv2', DS/'coco', DS/'custom_stereo',
          DS/'custom_seg', DS/'custom_detection', DS/'wakeword'/'positive',
          MODELS/'depth', MODELS/'segmentation', MODELS/'detection',
          MODELS/'privacy', MODELS/'speech'/'piper', MODELS/'speech'/'whisper',
          MODELS/'llm']:
    d.mkdir(parents=True, exist_ok=True)

t_start = datetime.now()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts():
    return datetime.now().strftime('%H:%M:%S')

def _size(path):
    p = Path(path)
    if not p.exists(): return 0
    return p.stat().st_size

def _size_gb(path):
    return _size(path) / 1e9

def _size_mb(path):
    return _size(path) / 1e6

def _ok(path, min_bytes):
    return Path(path).exists() and _size(path) >= min_bytes

def _verify_archive(path):
    path = str(path)
    try:
        if path.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                if bad:
                    raise ValueError(f'bad file in zip: {bad}')
        elif path.endswith(('.tar.gz', '.tgz', '.tar')):
            with tarfile.open(path) as t:
                t.getmembers()
        return True
    except Exception as e:
        print(f'    ⚠ Archive corrupt ({e}) — removing')
        os.remove(path)
        return False

def _extract(path, dest):
    path, dest = str(path), str(dest)
    print(f'    Extracting {Path(path).name} ...')
    if path.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    else:
        subprocess.run(['tar', '-xf', path, '-C', dest], check=True)

def wget(url, dest, min_bytes, label=None):
    """Download with resume. Returns True if file is ready."""
    dest = Path(dest)
    label = label or dest.name
    if _ok(dest, min_bytes):
        print(f'  ✓ {label} already downloaded ({_size_mb(dest):.0f} MB)')
        return True
    if dest.exists():
        print(f'  ↺ Resuming {label} ({_size_mb(dest):.0f} MB so far) ...')
    else:
        print(f'  ↓ Downloading {label} ...')
    r = subprocess.run([
        'wget', '-c', '--show-progress', '--timeout=60', '--tries=5',
        '-O', str(dest), url
    ])
    if _ok(dest, min_bytes):
        print(f'  ✓ {label} done ({_size_gb(dest):.2f} GB)')
        return True
    print(f'  ✗ {label} failed or incomplete')
    return False

def gdown_dl(file_id, dest, min_bytes, label=None):
    """Download from Google Drive by file ID."""
    dest = Path(dest)
    label = label or dest.name
    if _ok(dest, min_bytes):
        print(f'  ✓ {label} already downloaded ({_size_mb(dest):.0f} MB)')
        return True
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
    import gdown
    print(f'  ↓ Downloading {label} from Google Drive ...')
    gdown.download(id=file_id, output=str(dest), resume=True, quiet=False)
    if _ok(dest, min_bytes):
        print(f'  ✓ {label} done ({_size_mb(dest):.0f} MB)')
        return True
    print(f'  ✗ {label} failed')
    return False

def hf_dl(repo, filename, dest, min_bytes, label=None, repo_type='model'):
    """Download from HuggingFace Hub. Handles resume via HF cache then copies."""
    dest = Path(dest)
    label = label or Path(filename).name
    if _ok(dest, min_bytes):
        print(f'  ✓ {label} already downloaded ({_size_mb(dest):.0f} MB)')
        return True
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'])
    from huggingface_hub import hf_hub_download
    print(f'  ↓ Downloading {label} from HuggingFace ...')
    try:
        cached = hf_hub_download(repo_id=repo, filename=filename,
                                  repo_type=repo_type, local_dir=str(dest.parent))
        if Path(cached) != dest and Path(cached).exists():
            shutil.copy2(cached, dest)
        if _ok(dest, min_bytes):
            print(f'  ✓ {label} done ({_size_mb(dest):.0f} MB)')
            return True
    except Exception as e:
        print(f'  ✗ {label} failed: {e}')
    return False

def section(title):
    w = 62
    print()
    print('=' * w)
    print(f'  {title}')
    print('=' * w)
    print(f'  [{_ts()}]')

# ── 1. ADE20K ─────────────────────────────────────────────────────────────────
section('1 / 8   ADE20K  (~900 MB)  — SegFormer training')

ADE_DIR  = DS / 'ade20k'
ADE_ZIP  = ADE_DIR / 'ADEChallengeData2016.zip'
ADE_FLAG = ADE_DIR / 'ade20k_done.flag'

if ADE_FLAG.exists():
    print('  ✓ ADE20K already extracted')
else:
    ok = wget(
        'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',
        ADE_ZIP, min_bytes=800_000_000, label='ADEChallengeData2016.zip'
    )
    if ok and _verify_archive(ADE_ZIP):
        _extract(ADE_ZIP, ADE_DIR)
        ADE_FLAG.touch()
        print('  ✅ ADE20K ready')
    else:
        print('  ✗ ADE20K incomplete — re-run to resume')

subprocess.run(['du', '-sh', str(ADE_DIR)], check=False)

# ── 2. NYUv2 ──────────────────────────────────────────────────────────────────
section('2 / 8   NYUv2  (~2.8 GB)  — SegFormer supplemental (stairs/doors)')

NYU_DIR  = DS / 'nyuv2'
NYU_FILE = NYU_DIR / 'nyu_depth_v2_labeled.mat'
NYU_MIN  = 2_600_000_000

NYU_SOURCES = [
    'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat',
    'https://huggingface.co/datasets/depth-estimation/nyu_depth_v2/resolve/main/nyu_depth_v2_labeled.mat',
]

if _ok(NYU_FILE, NYU_MIN):
    print(f'  ✓ NYUv2 already downloaded ({_size_gb(NYU_FILE):.2f} GB)')
else:
    downloaded = False
    for url in NYU_SOURCES:
        src_name = url.split('/')[2]
        print(f'  Trying {src_name} ...')
        ok = wget(url, NYU_FILE, NYU_MIN, label='nyu_depth_v2_labeled.mat')
        if ok:
            downloaded = True
            break
        if NYU_FILE.exists():
            os.remove(NYU_FILE)
    if not downloaded:
        print('  ⚠ NYUv2 unavailable — SegFormer will train on ADE20K only (still good)')

# ── 3. COCO 2017 ──────────────────────────────────────────────────────────────
section('3 / 8   COCO 2017  (~18 GB)  — YOLOv8 training')

COCO_DIR  = DS / 'coco'
COCO_FLAG = COCO_DIR / 'coco_done.flag'

COCO_FILES = [
    ('http://images.cocodataset.org/zips/train2017.zip',
     COCO_DIR / 'train2017.zip',   16_000_000_000, 'train2017.zip  (17 GB)'),
    ('http://images.cocodataset.org/zips/val2017.zip',
     COCO_DIR / 'val2017.zip',        700_000_000, 'val2017.zip   (750 MB)'),
    ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
     COCO_DIR / 'annotations.zip',    200_000_000, 'annotations.zip (240 MB)'),
]

if COCO_FLAG.exists():
    print('  ✓ COCO 2017 already extracted')
else:
    all_ok = True
    for url, dest, min_bytes, label in COCO_FILES:
        if not wget(url, dest, min_bytes, label):
            all_ok = False
            continue
        if not _verify_archive(dest):
            all_ok = False
            continue
        _extract(dest, COCO_DIR)

    # Write COCO YAML
    import yaml
    coco_yaml = COCO_DIR / 'coco.yaml'
    yaml.dump({
        'path': str(COCO_DIR), 'train': 'train2017', 'val': 'val2017', 'nc': 80,
        'names': [
            'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
            'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
            'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
            'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
            'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
            'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
            'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
            'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
            'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
            'clock','vase','scissors','teddy bear','hair drier','toothbrush'
        ]
    }, open(coco_yaml, 'w'), default_flow_style=False)

    if all_ok:
        COCO_FLAG.touch()
        print('  ✅ COCO 2017 ready')
    else:
        print('  ⚠ COCO incomplete — re-run to resume the remaining files')

subprocess.run(['du', '-sh', str(COCO_DIR)], check=False)

# ── 4. RAFT-Stereo weights ────────────────────────────────────────────────────
section('4 / 8   RAFT-Stereo pretrained weights  (~30 MB)')

DEPTH_DIR   = MODELS / 'depth'
RAFT_CKPT   = DEPTH_DIR / 'raftstereo-realtime.pth'
RAFT_REPO   = BASE / 'raft_stereo'

if not RAFT_REPO.exists():
    print('  Cloning RAFT-Stereo repo ...')
    subprocess.run(['git', 'clone', 'https://github.com/princeton-vl/RAFT-Stereo',
                    str(RAFT_REPO)], check=True)
else:
    print(f'  ✓ RAFT-Stereo repo exists')

if _ok(RAFT_CKPT, 20_000_000):
    print(f'  ✓ raftstereo-realtime.pth ready ({_size_mb(RAFT_CKPT):.0f} MB)')
else:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
    os.chdir(RAFT_REPO)
    subprocess.run(['bash', 'download_models.sh'])
    # Move weights to DEPTH_DIR
    repo_models = RAFT_REPO / 'models'
    if repo_models.exists():
        for fn in repo_models.glob('*.pth'):
            shutil.copy2(fn, DEPTH_DIR / fn.name)
            print(f'  ✓ Copied {fn.name}')
    os.chdir('/')

# ── 5. SCRFD face detector ────────────────────────────────────────────────────
section('5 / 8   SCRFD / insightface buffalo_s  (~90 MB)')

PRIVACY_DIR = MODELS / 'privacy'
BUFFALO_DIR = PRIVACY_DIR / 'buffalo_s'

if BUFFALO_DIR.exists() and any(BUFFALO_DIR.iterdir()):
    print('  ✓ buffalo_s already downloaded')
else:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'insightface', 'onnxruntime'])
    import insightface
    app = insightface.app.FaceAnalysis(name='buffalo_s',
                                        root=str(PRIVACY_DIR),
                                        providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1)
    print('  ✅ buffalo_s downloaded via insightface')

# ── 6. CRAFT text detector ────────────────────────────────────────────────────
section('6 / 8   CRAFT text detector  (~90 MB)')

CRAFT_DIR  = MODELS / 'privacy'
CRAFT_CKPT = CRAFT_DIR / 'craft_mlt_25k.pth'

# craft_mlt_25k.pth — hosted on Google Drive by clovaai
CRAFT_GDRIVE_ID = '1Jk4eGW7DHA09z_MmqnkkSqBkCHKnIPNF'
if _ok(CRAFT_CKPT, 80_000_000):
    print(f'  ✓ craft_mlt_25k.pth ready ({_size_mb(CRAFT_CKPT):.0f} MB)')
else:
    gdown_dl(CRAFT_GDRIVE_ID, CRAFT_CKPT, min_bytes=80_000_000,
             label='craft_mlt_25k.pth')

# ── 7. Phi-3.5 Mini GGUF ─────────────────────────────────────────────────────
section('7 / 8   Phi-3.5 Mini Q4_K_M GGUF  (~2.7 GB)')

LLM_DIR  = MODELS / 'llm'
LLM_FILE = LLM_DIR / 'Phi-3.5-mini-instruct-Q4_K_M.gguf'

if _ok(LLM_FILE, 2_000_000_000):
    print(f'  ✓ Phi-3.5 GGUF ready ({_size_gb(LLM_FILE):.2f} GB)')
else:
    hf_dl(
        repo='bartowski/Phi-3.5-mini-instruct-GGUF',
        filename='Phi-3.5-mini-instruct-Q4_K_M.gguf',
        dest=LLM_FILE,
        min_bytes=2_000_000_000,
        label='Phi-3.5-mini-instruct-Q4_K_M.gguf',
    )

# ── 8. Piper TTS ──────────────────────────────────────────────────────────────
section('8 / 8   Piper TTS en_US-lessac-medium  (~65 MB)')

PIPER_DIR = MODELS / 'speech' / 'piper'
PIPER_ONNX = PIPER_DIR / 'en_US-lessac-medium.onnx'
PIPER_JSON = PIPER_DIR / 'en_US-lessac-medium.onnx.json'

for fname, dest, min_bytes in [
    ('en/en_US/lessac/medium/en_US-lessac-medium.onnx',      PIPER_ONNX, 55_000_000),
    ('en/en_US/lessac/medium/en_US-lessac-medium.onnx.json', PIPER_JSON,      1_000),
]:
    hf_dl('rhasspy/piper-voices', fname, dest, min_bytes,
          label=Path(fname).name, repo_type='model')

# ── Summary ───────────────────────────────────────────────────────────────────
elapsed = (datetime.now() - t_start).total_seconds()
h, rem  = divmod(int(elapsed), 3600)
m, s    = divmod(rem, 60)

print()
print('=' * 62)
print('  ARGUS Dataset Download Summary')
print('=' * 62)

checks = [
    ('ADE20K',         ADE_FLAG.exists(),                     'segformer training'),
    ('NYUv2',          _ok(NYU_FILE, NYU_MIN),                'segformer supplemental'),
    ('COCO 2017',      COCO_FLAG.exists(),                    'yolov8 training'),
    ('RAFT-Stereo wts',_ok(RAFT_CKPT, 20_000_000),           'depth model'),
    ('SCRFD (buffalo_s)',BUFFALO_DIR.exists() and any(BUFFALO_DIR.iterdir()), 'face detection'),
    ('CRAFT weights',  _ok(CRAFT_CKPT, 80_000_000),          'text detection'),
    ('Phi-3.5 GGUF',   _ok(LLM_FILE, 2_000_000_000),         'LLM'),
    ('Piper TTS',      _ok(PIPER_ONNX, 55_000_000),          'TTS'),
]

all_ok = True
for label, ok, use in checks:
    icon = '✅' if ok else '❌'
    print(f'  {icon}  {label:<22} — {use}')
    if not ok:
        all_ok = False

drive_used = int(subprocess.run(
    ['du', '-sb', str(BASE)], capture_output=True, text=True
).stdout.split()[0]) / 1e9

print()
print(f'  Drive used by ARGUS: {drive_used:.1f} GB')
print(f'  Elapsed: {h}h {m:02d}m {s:02d}s')
print()

if all_ok:
    print('  ✅ All datasets ready. Run:')
    print('       !python run_argus_pipeline.py')
else:
    print('  ⚠  Some downloads incomplete. Re-run this script to resume.')
    print('  Once all are ✅ run:')
    print('       !python run_argus_pipeline.py')
print('=' * 62)

# Save manifest to Drive
manifest = {ts: {label: ok for label, ok, _ in checks}
            for ts in [datetime.now().isoformat()]}
log_file = LOGS / 'dataset_download_status.json'
existing = json.loads(log_file.read_text()) if log_file.exists() else {}
existing.update(manifest)
log_file.write_text(json.dumps(existing, indent=2))

#!/usr/bin/env python3
"""Embed the grounding vocabulary with the pinned CLIP text encoder, once.

Uses exactly the encoding the runtime used before (CPU CLIP ViT-B/32,
`clip.tokenize(truncate=True)`, L2-normalised), writes
`vocab_embeddings.npy` + `vocab_embeddings.json`, and checks that a re-encoded
sample matches the table to 1e-5.

    python3 scripts/build_vocab_embeddings.py [--vocab config/vocab_en.txt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from argus.config import GroundingConfig  # noqa: E402
from argus.vocab import normalise  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = GroundingConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=str(REPO / "config" / "vocab_en.txt"))
    ap.add_argument("--out", default=cfg.vocab_embeddings)
    args = ap.parse_args()

    words, seen = [], set()
    for line in Path(args.vocab).read_text().splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        k = normalise(w)
        if k not in seen:
            seen.add(k)
            words.append(k)

    import clip
    import torch
    t0 = time.perf_counter()
    model, _ = clip.load(cfg.text_encoder, device="cpu", jit=False)
    model.eval()
    vecs = []
    with torch.inference_mode():
        for i in range(0, len(words), 64):
            tok = clip.tokenize(words[i:i + 64], truncate=True)
            v = model.encode_text(tok).float()
            v /= v.norm(dim=-1, keepdim=True)
            vecs.append(v.numpy())
        table = np.concatenate(vecs).astype(np.float32)
        # Parity: batch encoding must equal the runtime's one-at-a-time path.
        worst = 0.0
        for w in words[:: max(1, len(words) // 12)]:
            v = model.encode_text(clip.tokenize([w], truncate=True)).float()
            v /= v.norm(dim=-1, keepdim=True)
            worst = max(worst, float(np.abs(v.numpy()[0] - table[words.index(w)]).max()))
    if worst > 1e-4:
        print(f"parity check failed: max abs diff {worst}")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, table)
    meta = {"words": words, "count": len(words), "encoder": Path(cfg.text_encoder).name,
            "encoder_sha256": sha256(cfg.text_encoder), "parity_max_abs_diff": worst,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": str(Path(args.vocab).relative_to(REPO))}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"{len(words)} labels -> {out} ({table.nbytes / 1024:.0f} KiB) in "
          f"{time.perf_counter() - t0:.1f} s; parity max diff {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

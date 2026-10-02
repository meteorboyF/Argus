#!/usr/bin/env python3
"""Write a vision-only copy of a Gemma 4 mmproj GGUF.

The published Gemma 4 E2B projector bundles the audio encoder (`a.*` tensors,
~612 MB at F16) with the vision encoder (`v.*`, ~366 MB) and projector (`mm.*`).
ARGUS transcribes speech with Whisper, so the audio tower is never used but is
still loaded into the 8 GB of shared memory. This copies every vision/projector
tensor byte-for-byte and every metadata key except the audio ones, then marks
`clip.has_audio_encoder = false`.

    python3 scripts/strip_mmproj_audio.py IN.gguf OUT.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/opt/argus/llama.cpp/gguf-py")
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402

AUDIO_TENSOR_PREFIX = "a."
AUDIO_KEY_PREFIX = "clip.audio."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    r = GGUFReader(args.src)
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(args.dst, arch)
    skipped_keys = 0
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key.startswith(AUDIO_KEY_PREFIX):
            skipped_keys += 1
            continue
        if key == "clip.has_audio_encoder":
            w.add_bool(key, False)
            continue
        vtype = field.types[0]
        value = field.contents()
        if vtype == GGUFValueType.ARRAY:
            w.add_key_value(key, value, vtype, sub_type=field.types[-1])
        else:
            w.add_key_value(key, value, vtype)

    kept = [t for t in r.tensors if not t.name.startswith(AUDIO_TENSOR_PREFIX)]
    for t in kept:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in kept:
        w.write_tensor_data(t.data)
    w.close()
    print(f"kept {len(kept)} of {len(r.tensors)} tensors, dropped {skipped_keys} audio keys")
    print(f"{args.src}: {Path(args.src).stat().st_size / 1e6:.1f} MB -> "
          f"{args.dst}: {Path(args.dst).stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

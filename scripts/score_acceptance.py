#!/usr/bin/env python3
"""Score private observations; report contains no reference/hypothesis text.

See docs/ACCEPTANCE.md for schemas. Keep the input outside the repository.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argus.acceptance import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['reading', 'seat', 'wake', 'demo'])
    parser.add_argument('observations', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = score(args.kind, json.loads(args.observations.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

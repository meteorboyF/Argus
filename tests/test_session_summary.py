"""Session log -> study metrics."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "summarise_session", Path(__file__).parents[1] / "scripts" / "summarise_session.py")
ss = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ss
_SPEC.loader.exec_module(ss)


def _events():
    e = [{"t": 0.0, "event": "system", "text": "ARGUS ready"}]
    e += [{"t": 10.0, "event": "hazard_announced", "rule": "body", "level": "WARN",
           "distance_m": 2.3},
          {"t": 30.0, "event": "hazard_announced", "rule": "body", "level": "DANGER",
           "distance_m": 1.1},
          {"t": 50.0, "event": "hazard_announced", "rule": "head", "level": "DANGER",
           "distance_m": 1.6},
          {"t": 70.0, "event": "approach_announced", "level": "WARN", "ttc_s": 2.8},
          {"t": 80.0, "event": "query_end", "total_s": 2.1},
          {"t": 90.0, "event": "query_end", "total_s": 3.5},
          {"t": 100.0, "event": "seat_arrived", "duration_s": 24.0},
          {"t": 110.0, "event": "speech_preempted"},
          {"t": 120.0, "event": "skew_drop", "total": 3}]
    return e


def test_summary_without_truth():
    r = ss.summarise(_events())
    assert r["duration_min"] == 2.0
    assert r["alarms"]["body"] == {"count": 2, "per_min": 1.0, "danger": 1}
    assert r["alarms_per_min_total"] == 2.0
    assert r["lead"]["ttc_s_at_announcement"]["median"] == 2.8
    assert r["queries"]["n"] == 2 and r["seat"]["arrived"]["median"] == 24.0
    assert r["speech"]["preemptions"] == 1 and r["skew_drops"] == 3


def test_accuracy_against_annotated_truth():
    truth = [{"start": 9.5, "end": 12.0, "rule": "body"},       # caught at 10
             {"start": 40.0, "end": 45.0, "rule": "body"},      # missed
             {"start": 49.0, "end": 52.0, "rule": "head"}]      # caught at 50
    acc = ss.summarise(_events(), truth)["accuracy"]
    assert acc["body"]["precision"] == 0.5 and acc["body"]["recall"] == 0.5
    assert acc["body"]["false_alarms_per_min"] == 0.5 and acc["body"]["missed"] == 1
    assert acc["head"]["precision"] == 1.0 and acc["head"]["recall"] == 1.0


def test_torn_last_line_is_tolerated(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in _events()) + '\n{"t": 13')
    assert len(ss.load(str(p))) == len(_events())

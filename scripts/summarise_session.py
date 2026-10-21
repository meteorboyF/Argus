#!/usr/bin/env python3
"""Study metrics from an ARGUS session log (JSONL from SessionLogger).

    python3 scripts/summarise_session.py reports/sessions/2026-10-21-1400.jsonl \
        [--truth truth.json] [--out summary.json]

Without ground truth: alarms per minute (all announced hazards), warning
lead time (distance / TTC at announcement), task times (reading, seat
finding, queries) and latency percentiles.

With --truth (a JSON list of {"start": s, "end": s, "rule": "body|head|drop|
approach"} intervals, seconds from the session start, annotated from the demo
video): per-rule precision, recall, and false alarms per minute. An
announcement is a true positive if it falls inside an interval of the same
rule (or up to 1 s before it starts).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: str) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass        # a torn last line from a crash is not fatal
    out.sort(key=lambda e: e["t"])
    return out


def pct(values) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": round(float(np.median(v)), 3),
            "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3)}


def summarise(events: list[dict], truth: list[dict] | None = None) -> dict:
    if not events:
        return {"events": 0}
    t0, t1 = events[0]["t"], events[-1]["t"]
    minutes = max((t1 - t0) / 60.0, 1e-6)
    hz = [e for e in events if e["event"] in ("hazard_announced", "approach_announced")]
    by_rule = defaultdict(list)
    for e in hz:
        by_rule[e.get("rule", "approach")].append(e)
    res = {
        "duration_min": round(minutes, 2),
        "events": len(events),
        "alarms": {r: {"count": len(v), "per_min": round(len(v) / minutes, 2),
                       "danger": sum(1 for x in v if x.get("level") == "DANGER")}
                   for r, v in by_rule.items()},
        "alarms_per_min_total": round(len(hz) / minutes, 2),
        "lead": {
            "distance_m_at_announcement": pct(e.get("distance_m") for e in hz
                                              if e["event"] == "hazard_announced"),
            "ttc_s_at_announcement": pct(e.get("ttc_s") for e in hz
                                         if e["event"] == "approach_announced"),
        },
        "queries": pct(e.get("total_s") for e in events if e["event"] == "query_end"),
        "reading": {"capture_s": pct(e.get("capture_s") for e in events if e["event"] == "reading"),
                    "ocr_s": pct(e.get("ocr_s") for e in events if e["event"] == "reading")},
        "seat": {"arrived": pct(e.get("duration_s") for e in events if e["event"] == "seat_arrived"),
                 "abandoned": sum(1 for e in events if e["event"] == "seat_abandoned")},
        "speech": {"preemptions": sum(1 for e in events if e["event"] == "speech_preempted"),
                   "stale_safety_dropped": sum(1 for e in events
                                               if e["event"] == "speech_stale_drop"),
                   "queue_wait_s": pct(e.get("queued_s") for e in events
                                       if e["event"] == "speech_start")},
        "skew_drops": max([e.get("total", 0) for e in events if e["event"] == "skew_drop"],
                          default=0),
        "safety_paused": sum(1 for e in events if e["event"] == "safety_paused"),
    }
    if truth is not None:
        res["accuracy"] = score(hz, truth, t0, minutes)
    return res


def score(hz: list[dict], truth: list[dict], t0: float, minutes: float) -> dict:
    out = {}
    rules = sorted({t["rule"] for t in truth} | {e.get("rule", "approach") for e in hz})
    for rule in rules:
        intervals = [t for t in truth if t["rule"] == rule]
        alerts = [e["t"] - t0 for e in hz if e.get("rule", "approach") == rule]
        tp = sum(1 for a in alerts if any(i["start"] - 1.0 <= a <= i["end"] for i in intervals))
        hit = sum(1 for i in intervals if any(i["start"] - 1.0 <= a <= i["end"] for a in alerts))
        fp = len(alerts) - tp
        out[rule] = {"precision": round(tp / len(alerts), 3) if alerts else None,
                     "recall": round(hit / len(intervals), 3) if intervals else None,
                     "false_alarms_per_min": round(fp / minutes, 2),
                     "missed": len(intervals) - hit}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    truth = json.loads(Path(args.truth).read_text()) if args.truth else None
    res = summarise(load(args.log), truth)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

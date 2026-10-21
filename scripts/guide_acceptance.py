#!/usr/bin/env python3
"""Speak/display a deferred acceptance protocol, without claiming it passed.

Run separately from ARGUS to avoid competing speech queues. This rehearses the
protocol and writes a pending checklist; it does not start cameras or a walk.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argus.config import load_config
from argus.guide import Guide

PROTOCOLS = {
    'reading': [
        'Stay seated. Hold an English page at arm length. First run Argus doctor focus.',
        'Ask Argus, what is the name of this book? Check both title and author against the cover.',
        'Ask Argus, read this. Measure capture to the first spoken word. Save a private reference transcription.',
        'Try next page, what is this page about, stop, repeat, and a long page.',
        'Test backlight, glare, upside down text, two columns, a cut off page, a hand, no text, and sensitive numbers.',
    ],
    'seat': [
        'Wait until fast loop and calibration acceptance pass before walking with guidance.',
        'Plan ten supervised trials across three room layouts. Record success and seconds for each.',
        'Include no seat, two seats, an occupied chair, a seat behind you, and a lost target. Never approach a person.',
    ],
    'wake': [
        'Plan fifty real voice attempts saying Argus followed by a request. Record each wake as true or false.',
        'Record false wakes over at least two hours of normal room sound including Bangla, English, and television.',
    ],
    'demo': [
        'Wait until calibration and approach warning acceptance pass.',
        'Record two minutes with startup summary, find object, reading, and an approach warning interrupting speech.',
        'Keep the video local. Review and commit only the masked session summary.',
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=PROTOCOLS)
    parser.add_argument('--seconds-per-step', type=float, default=12)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds_per_step <= 0:
        parser.error('seconds per step must be positive')
    with Guide(load_config().speech, title='ARGUS acceptance preparation') as guide:
        try:
            steps = PROTOCOLS[args.kind]
            for i, step in enumerate(steps):
                guide.say(step, force=True)
                deadline = time.monotonic()+args.seconds_per_step
                while time.monotonic() < deadline or guide.speaking:
                    if guide.show(progress=(i+1)/len(steps)) == 27:
                        raise KeyboardInterrupt
                    time.sleep(.05)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({'kind':args.kind, 'status':'partial',
                                           'protocol_rehearsed':True, 'device_trials':False}, indent=2)+'\n')
            guide.done('Protocol reviewed. Device acceptance is still pending.')
        except KeyboardInterrupt:
            guide.failed('Protocol rehearsal cancelled. No acceptance recorded.')
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

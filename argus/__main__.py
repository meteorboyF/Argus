"""ARGUS runtime entry point.

Usage (on the Jetson, after setup):
    python -m argus run                 # full two-speed runtime
    python -m argus run --no-audio      # fast loop only (no mic/speaker)
    python -m argus query "what is in front of me?"   # one slow-path turn
    python -m argus selftest            # check imports, models, camera, server
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config


def _cmd_run(args):
    from .orchestrator import Orchestrator
    cfg = load_config(args.config)
    orch = Orchestrator(cfg, enable_audio=not args.no_audio)
    orch.run()


def _cmd_query(args):
    from .orchestrator import Orchestrator
    cfg = load_config(args.config)
    orch = Orchestrator(cfg, enable_audio=not args.no_audio)
    try:
        orch.start_fast_loop()
        import time
        time.sleep(1.0)  # let the fast loop settle after its verified first map
        orch.handle_query(args.text)
        # speak() is non-blocking — drain the speaker before tearing down, or the
        # process exits mid-sentence.
        orch.speaker.wait_until_idle(timeout=60.0)
    finally:
        orch.stop()


def _cmd_selftest(args):
    from .selftest import run_selftest
    sys.exit(0 if run_selftest(load_config(args.config)) else 1)


def _cmd_preview(args):
    from .preview import run_preview
    run_preview(load_config(args.config), seconds=args.seconds, describe=args.describe)


def _cmd_baseline(args):
    from .diagnostics import collect_baseline, print_report, write_report
    cfg = load_config(args.config)
    report = collect_baseline(cfg)
    print_report(report)
    if args.output:
        write_report(report, args.output)
        print(f"JSON report: {args.output}")
    sys.exit(0 if report["production_ready"] else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="argus", description="ARGUS smart-glasses runtime")
    p.add_argument("--config", default=None, help="path to argus.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the full two-speed runtime")
    pr.add_argument("--no-audio", action="store_true", help="fast loop only, no speech")
    pr.set_defaults(func=_cmd_run)

    pq = sub.add_parser("query", help="run one slow-path interaction")
    pq.add_argument("text", help="the question to ask")
    pq.add_argument("--no-audio", action="store_true", help="print the answer instead of speaking")
    pq.set_defaults(func=_cmd_query)

    ps = sub.add_parser("selftest", help="check environment, models, camera, llama server")
    ps.set_defaults(func=_cmd_selftest)

    pp = sub.add_parser("preview", help="show normalized left/right/wide camera feeds")
    pp.add_argument("--seconds", type=float, default=0,
                    help="close after N seconds (default: run until Q/Esc)")
    pp.add_argument("--describe", action="store_true",
                    help="ask ARGUS to describe one privacy-gated center frame")
    pp.set_defaults(func=_cmd_preview)

    pb = sub.add_parser("baseline", help="read-only Jetson environment baseline")
    pb.add_argument("--output", default=None, help="optional JSON report path")
    pb.set_defaults(func=_cmd_baseline)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

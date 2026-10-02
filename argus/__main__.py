"""ARGUS runtime entry point.

Usage (on the Jetson, after setup):
    python -m argus run                 # full two-speed runtime
    python -m argus run --no-audio      # fast loop only (no mic/speaker)
    python -m argus query "what is in front of me?"   # one slow-path turn
    python -m argus selftest            # check imports, models, camera, server
    python -m argus doctor              # rig bring-up table + spoken summary
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config


def _cmd_run(args):
    from .orchestrator import Orchestrator
    cfg = load_config(args.config)
    orch = Orchestrator(cfg, enable_audio=not args.no_audio,
                        enable_mic=not (args.no_audio or args.no_mic))
    orch.run(dashboard=args.dashboard, fullscreen=not args.windowed,
             ask_file=args.ask_file)


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


def _cmd_doctor(args):
    from . import doctor
    cfg = load_config(args.config)
    stamp = __import__("time").strftime("%Y-%m-%d-%H%M")
    speak = not args.quiet
    if args.snap:
        doctor.run_snap(cfg, seconds=args.seconds, speak=speak)
        return
    if args.skew_test:
        rep = doctor.run_skew_test(cfg, seconds=args.seconds * 2, stereo_fps=args.stereo_fps,
                                   speak=speak)
        doctor.write_json(rep, f"skew-test-{stamp}.json")
        sys.exit(0 if "error" not in rep else 1)
    if args.bandwidth:
        rep = doctor.run_bandwidth(cfg, speak=speak)
        doctor.write_json(rep, f"usb-bandwidth-{stamp}.json")
        return
    rep = doctor.run_doctor(cfg, seconds=args.seconds, speak=speak)
    doctor.write_json(rep, f"doctor-{stamp}.json")
    sys.exit(0 if rep["all_ok"] else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="argus", description="ARGUS smart-glasses runtime")
    p.add_argument("--config", default=None, help="path to argus.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the full two-speed runtime")
    pr.add_argument("--no-audio", action="store_true", help="fast loop only, no speech")
    pr.add_argument("--no-mic", action="store_true",
                    help="speak answers but do not listen; ask by typing in the dashboard")
    pr.add_argument("--dashboard", action="store_true",
                    help="show the live monitor view (cameras, depth, detections, speech)")
    pr.add_argument("--windowed", action="store_true", help="dashboard in a window, not fullscreen")
    pr.add_argument("--ask-file", default=None, metavar="PATH",
                    help="also accept questions appended as lines to this file "
                         "(e.g. echo 'find the monitor' >> PATH)")
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

    pd = sub.add_parser("doctor", help="rig bring-up checks with a spoken summary")
    pd.add_argument("--seconds", type=float, default=10.0, help="measurement window")
    pd.add_argument("--snap", action="store_true",
                    help="guided worn-orientation snapshots to /tmp/argus_snap/")
    pd.add_argument("--skew-test", action="store_true",
                    help="true stereo exposure offset from an on-screen timecode")
    pd.add_argument("--bandwidth", action="store_true",
                    help="stereo fps matrix with the wide camera closed vs streaming")
    pd.add_argument("--stereo-fps", type=int, default=None, help="override stereo fps (skew test)")
    pd.add_argument("--quiet", action="store_true", help="no speech")
    pd.set_defaults(func=_cmd_doctor)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

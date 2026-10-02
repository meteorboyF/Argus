# ARGUS progress

Updated: 2026-10-02. Short, current, and honest. `STATUS.md` holds the
per-component table; this file says what happened last and what is blocking.

## What was done last (night of 2026-09-29, commits a4b6eb5..806b8ba)

- **Monitor dashboard.** `python3 -m argus run --dashboard --no-mic --ask-file PATH`
  shows all three camera feeds, the depth map, a top-down corridor/SLAM map, a
  status panel and the spoken conversation on the attached monitor. Questions
  can be typed in the window or appended to the ask file. Verified live.
- **Slow loop speed.** A typed "find the monitor" round-trips in 1.8 s
  (privacy 0.17 s, grounding 0.72 s, Gemma final turn 0.91 s). Gemma 4 E2B runs
  in llama-server on CUDA with `-ngl 99` and flash attention. The CLIP text
  encoder warms at startup instead of on the first question.
- **Fast loop: incoming-vehicle rule.** Per-zone closing speed and
  time-to-collision in `argus/safety.py` (WARN at 3 s, DANGER at 1.5 s).
  Laya was evaluated at the user's request and rejected for this path; see
  `DECISION_LOG.md`.
- **SLAM and navigation.** Stereo visual odometry (`argus/slam.py`) and
  corridor guidance (`argus/navigation.py`) with synthetic tests. Both switch
  on automatically once a stereo calibration file exists.
- **Cameras.** Per-camera capture threads with real arrival timestamps, role
  binding by hub port across the USB 3 / USB 2 companion buses, live link
  speed on the dashboard. Two cameras that had fallen back to USB 2 were
  brought back to 5 Gbit/s by reseating their screw-lock USB-C plugs.
- **Calibration coaching.** `scripts/calibrate_stereo.py --coach` runs
  fullscreen with large on-screen instructions for whoever holds the board,
  finishes on its own after 15 diverse views, and shows the result.
- 77 unit tests pass (`pytest -q tests/ --ignore=tests/test_speech_priority.py`).

## What is halting progress

1. **No stereo calibration file.** `/opt/argus/config/stereo_calib.npz` does
   not exist. Until it does, metric hazard speech, the approach rule, SLAM and
   corridor guidance all stay off by design. Everything downstream waits on
   this one physical step.
2. **Camera orientation is unresolved.** During the first calibration attempt
   both stereo pictures were upside down (the holder's head at the bottom of
   the frame), whereas earlier that night, with the rig resting on the desk,
   they were upright. Calibration bakes in the orientation, so the rig must be
   held in its worn orientation and the config rotations (currently 0) must
   match it before any views are captured. Open question to the user: with the
   rig held as worn, is the picture upright or inverted?
3. **Chessboard square size is unknown.** The board is a folding 8x8 chess
   board (7x7 inner corners). The calibrator needs the measured square size in
   millimetres to set metric scale. A placeholder run saved nothing; the
   attempt was stopped before any capture because of item 2.
4. **Only one camera is plugged in right now** (one B0495 on hub port 4).
   Calibration needs both stereo cameras, and the demo needs all three.
5. **USB-C screw-lock plugs are the weak point.** A plug seated at a slight
   angle links at USB 2 (480 Mbit/s): 10 fps and multi-second stalls. The
   dashboard camera line turns amber with "reseat!" when this happens. Tighten
   both screws and strain-relieve the cables to the frame.
6. **No USB microphone connected**, so wake word and speech-to-text remain
   unvalidated. The dashboard's typed questions cover the demo meanwhile.
7. **Memory headroom.** The full stack with llama-server resident sits near
   6.7 GB of 7.6 GB RAM. Not blocking the demo, but there is little slack.

## Next steps, in order

1. Plug in both stereo cameras and the wide camera; confirm three "USB3" on
   the dashboard camera line.
2. Settle the worn orientation; set `left_rotation`/`right_rotation` in
   `config/argus.yaml` to 180 if the worn view is inverted.
3. Measure one square of the board in mm, then run
   `DISPLAY=:0 python3 scripts/calibrate_stereo.py --square-mm <N> --coach`
   and follow the on-screen text. Verify with
   `python3 scripts/calibrate_stereo.py --verify` against a tape measure at
   0.5, 1, 2 and 3 m.
4. Restart the dashboard and watch the approach rule, SLAM trail and corridor
   guidance come alive.

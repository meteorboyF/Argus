# Deferred device acceptance

Software build: 2026-10-21. Fardeen requested building now and physical testing
later. No new live acceptance is claimed. Calibration gates remain unchanged.
The prior automated T1 result does not certify the newer build.

## Headless service (T10)

`python3 scripts/install_user_service.py` installs the user unit and reloads
systemd without starting it. `--enable` enables future user sessions. It does
not enable lingering or change privileged files. At the chosen test time run
`systemctl --user start argus`; inspect `systemctl --user status argus` and
`/opt/argus/logs/runtime.log` / `llama-server.log`. Stop with
`systemctl --user stop argus`. Runtime startup already speaks “ARGUS ready”.
The supervisor stops its runtime before speaking a failure reason and exits
nonzero for bounded systemd restart. It reuses a healthy external model server
without killing it. Voice shutdown exits cleanly, so it stays shut down.
Boot before login requires existing user lingering; check `loginctl show-user
argus -p Linger`. If disabled, Fardeen must enable it separately when desired.
No service has been started or enabled by this build session.

## Reading / Bangla (T2, T8)

Default capture remains one post-settle still. Experiment explicitly with
`rig.capture_still(candidates=3)` versus 1; selection retains only the best
frame and scores downsampled sharpness plus usable exposure. It cannot repair
optical defocus or absent light. Compare latency/CER before changing defaults.
OCR retries a 180-degree image only when recognition has fewer than four
plausible characters. Two-column ordering requires a clear repeated gutter;
complex layouts remain unaccepted. Stop cancels coaching and discards an OCR
result that finishes after cancellation. Speech chunks are bounded.

`python3 scripts/evaluate_ocr.py /tmp/private-ocr.json --backend rapidocr
--out reports/ocr-English-DATE.json` scores offline recognition.
For Bangla select `--backend tesseract-ben`. The adapter requires installed
`ben+eng` language packs; it never installs packages or changes the runtime.
Fardeen can later install `tesseract-ocr-ben` using sudo. PaddleOCR-VL Bangla
GGUF integration remains pending model/backend evaluation; no GGUF is assumed
compatible or downloaded by this build.

The private manifest is a JSON list: `[{"image":"/absolute/sample.png",
"reference":"exact transcription"}]`. Keep source text and images outside git.
The report contains only numeric sample IDs, CER and OCR latency. It is not
capture-to-first-spoken-word acceptance or a resource benchmark.

## Scoring observations (T2, T5, T6, T9)

Run `python3 scripts/score_acceptance.py KIND /tmp/private-observations.json
--out reports/KIND-acceptance-DATE.json`. Exit 1 means partial. Source text is
never copied to the output. NFC normalization is the only CER normalization;
punctuation, case and spaces count. Empty references are errors.

Input fields:

- `reading`: `samples` list with `reference`, `hypothesis`, `language` (`eng`
  for English), `first_word_s`; `title_correct`, `author_correct`; `cases`
  booleans for `backlit`, `glare`, `upside_down`, `two_columns`, `cut_off`,
  `hand`, `no_text`, `long_page_controls`, `sensitive_numbers`.
- `seat`: `trials` list with `layout`, boolean `success`, nonnegative `seconds`;
  `cases` booleans for `no_seat`, `two_seats`, `occupied`, `behind`, `lost`.
- `wake`: `attempts` list of booleans, `ambient_hours`, integer `false_wakes`,
  `sounds` booleans for `bangla`, `english`, `tv`.
- `demo`: `seconds`, `recording_verified`, `segments` booleans for `startup`,
  `find_object`, `reading`, `approach_interrupt`.

Only record observed successes. The scorer trusts evaluator inputs; a passed
report requires evidence review before STATUS changes. It does not generate
truth labels, measure audio onset, or infer unobserved failures. Existing
`scripts/summarise_session.py --truth` scores the annotated walk; existing
calibration coach/verify tools retain their strict gate. T4 remains blocked
on T1/T3 acceptance. Do not start a walk from these preparation checklists.

`DISPLAY=:0 python3 scripts/guide_acceptance.py KIND --out
reports/KIND-protocol-DATE.json` speaks/displays a protocol rehearsal with
continuous screen feedback and DONE/FAILED. Run separately from ARGUS. It
writes partial even after a successful rehearsal; it does not execute trials.

## VLM comparison (T7)

`bench_vlm.py --variant-file /tmp/variants.json --variants NAME1,NAME2
--tasks /tmp/tasks.json --out reports/vlm-DATE.json` accepts custom variants.
Each variant supplies `model`, `mmproj`, `args` and optionally `tokens: true`,
`token_budget` (280 or 560), `cell` (architecture-specific image patch area
scale; do not assume Gemma's 48 applies to Qwen), or `max_side`.
Supply actual locally built/downloaded model paths; the benchmark does not
switch production weights. Repeat each model at both budgets. Model and
projector SHA-256 are written in the report. Use batch/microbatch 256 or a
measured alternative. Models are evaluated sequentially on benchmark port
8091. Stop the production server at test time if memory cannot hold both;
the tool does not kill someone else's server. Every frame is face gated and
OCR sensitive boxes are blurred before the model call; prompts/answers/expected
text are masked. Only one full frame is retained at a time. OCR privacy masking
still requires adversarial testing and does not guarantee all numbers are found.
`DISPLAY=:0 python3 scripts/collect_bakeoff_frames.py` guides the 30-frame
collection later, writes only privacy processed images, and checkpoints task
metadata after each frame. Stop ARGUS first so the collector owns the rig.
Use `--out-dir` and benchmark `--frames` for a different output directory.
The benchmark refuses a healthy production server on :8080 by default; the
explicit `--allow-concurrent-server` override requires an adequate memory budget.
Capture the planned 30 consented scenes/signs/pages later. Raw face collection,
automatic model downloads and an unmeasured model switch are not performed.

## Acceptance still required

T1 real-voice soak and fault tests; T2 focus, CER and actual speech onset;
T3 measured calibration and tape verification; T4 annotated walk/audio
preemption; T5 seat trials; T6 real recall/ambient recording; T7 frame/model
bake-off; T8 Bangla/model comparison; T9 recording; T10 real restart/boot/audio
checks. Keep every corresponding STATUS row partial until its device DoD passes.

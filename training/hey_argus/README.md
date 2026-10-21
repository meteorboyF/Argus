# "Hey Argus" wake word

ARGUS currently wakes on openWakeWord's built-in `hey_jarvis`. Those weights
are CC BY-NC-SA, and "Jarvis" is not the product's name. This folder trains a
custom **"Hey Argus"** model with openWakeWord's synthetic-speech pipeline:
Piper voices generate thousands of positive clips and the model learns from
those plus large negative feature sets. No recordings of real people are
needed.

**Target:** at most 0.5 false accepts per hour at 80% recall or better.

## Where to run it

Off-device: a CUDA PC or Google Colab, about 1–2 h on a T4. Do **not** run
training on the Jetson: it needs ~16 GB RAM and tens of GB of downloads.

## Steps

```bash
git clone https://github.com/dscripka/openWakeWord && cd openWakeWord
pip install -e '.[full]'
git clone https://github.com/rhasspy/piper-sample-generator
wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt \
  'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
# Negative data and augmentation sets, as in the openWakeWord training notebook:
#   mit_rirs/, audioset_16k/, fma/ (background audio),
#   openwakeword_features_ACAV100M_2000_hrs_16bit.npy,
#   validation_set_features.npy
# (download commands: openWakeWord notebooks/automatic_model_training.ipynb)
cp /path/to/Argus/training/hey_argus/hey_argus.yml .

python openwakeword/train.py --training_config hey_argus.yml --generate_clips
python openwakeword/train.py --training_config hey_argus.yml --augment_clips
python openwakeword/train.py --training_config hey_argus.yml --train_model
# -> hey_argus_model/hey_argus.onnx
```

## Install on the Jetson

```bash
scp hey_argus_model/hey_argus.onnx argus@<jetson>:/opt/argus/models/hey_argus.onnx
sha256sum /opt/argus/models/hey_argus.onnx      # record it in STATUS.md pins
```

Then in `/opt/argus/config/argus.yaml`:

```yaml
speech:
  wake_model: "/opt/argus/models/hey_argus.onnx"
  wake_threshold: 0.5      # tune with the false-accept test below
```

`argus/speech.py` already loads a custom `.onnx` path (`WakeWord`).

## Acceptance test (on the Jetson, with the headset mic)

1. **Recall:** say "Hey Argus" 50 times at normal, quiet and fast pace, from
   several speakers. Count the wakes. Must be at least 40 of 50.
2. **False accepts:** leave it running for 2 hours of normal room noise, with
   conversation in English and Bangla, a TV and street sound. Count the wakes.
   Must be at most 1 (0.5 per hour).
3. Record both numbers and the threshold in `DECISION_LOG.md`.

Until then, typed questions (dashboard / `--ask-file`) and `hey_jarvis` remain
the fallback.

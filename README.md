# Indic Voice Forensic Inspector

Detects AI-generated speech on Indian telephony channels, and says **which of six
vocoder families** produced it.

## What it does

| Tier | Question | Status |
|---|---|---|
| 0 | Does it carry a provenance watermark? | Optional, skipped cleanly if `audioseal` is absent |
| 1 | Real human, or AI-generated? | Trained head |
| 2 | Which of six decoder families made it? | Trained head |
| 3 | Is this a synthesis method we have never seen? | Mahalanobis, fitted after training |
| — | **Is this the person it claims to be?** | Pretrained ECAPA, no training needed |

The six families are HiFi-GAN, BigVGAN, Vocos, EnCodec, modern codecs (DAC / SNAC /
Mimi / Firefly) and legacy autoregressive. Plus bonafide, so seven classes.
`src/families.py` documents which systems fall into each and the acoustic tell for
every one.

## Documentation

| File | What |
|---|---|
| `HOW_IT_WORKS.md` | End-to-end architecture with line references |
| `CHANGES_FOR_HANS.md` | v0.3 → v1.1: every bug, every edit |
| `SPEAKER_VERIFICATION.md` | The speaker layer, and the 7 remaining tasks |
| `WORK_SPLIT.md` | Three-way detection: what Hans does, what we do |
| `ARCHITECTURE.md` | Original design document, unmodified |

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the full codec set you also need ffmpeg with `libopencore-amrnb`, `libgsm` and
`libopus`. Check what your machine can do:

```bash
python -c "from src.telephony_degrader import available_codecs; print(available_codecs())"
```

Missing codecs are reported, never silently skipped.

## Run the API

```bash
source venv/bin/activate
python server/api.py
```

- `GET /health` — is a model loaded, which codecs work
- `GET /v1/families` — the taxonomy
- `POST /v1/detect` — upload a clip to run the forensic pipeline

Audio is decoded in memory and never written to disk permanently.

## Run the dashboard

```bash
cd frontend
npm install
npm run dev
```

## Cross-session speaker verification

Answers a different question from the tiers above: *is this the person it claims to be?*
Catches a perfect clone that leaves no artifacts, and catches a human impersonator, which
no artifact detector ever will.

**Nothing is trained.** ECAPA-TDNN is pretrained on VoxCeleb; enrollment is a forward
pass. No extra dataset, no GPU hours.

```bash
pip install speechbrain

# Enrol from three genuine calls, on separate occasions
curl -X POST localhost:8000/v1/speakers/cfo_001/enroll \
  -F "files=@jan.wav" -F "files=@feb.wav" -F "files=@mar.wav" -F "display_name=A. Rao"

# Verify a live call, with context
curl -X POST localhost:8000/v1/analyze -F "file=@call.wav" \
  -F "speaker_id=cfo_001" -F "number_is_known=false" -F "is_privileged_action=true"
```

The store holds 192-dimension voiceprints and metadata. **No audio, ever** — asserted by
`test_store_holds_no_audio`.

Read `SPEAKER_VERIFICATION.md` before changing anything: the within-call stability signal
runs opposite to intuition, and "not enrolled" is deliberately not "rejected".

## Build data

```bash
python src/merge_datasets.py
python src/build_v1_dataset.py
python split_manifest.py
```

Splits are **speaker-disjoint by stable hash**, silence is trimmed identically for
both classes, and a JSON Lines manifest carries label, family, speaker and language.

## Train

**Always sanity-check first.** Five minutes here beats discovering a broken head
after eight hours:

```bash
python train.py --overfit-batch
```

Loss must fall below 0.05 on sixteen files. If it does not, stop — the label
mapping or the learning rate is wrong.

```bash
python train.py --data ./data --out ./checkpoints/vfd-v1
```

Defaults: learning rate 3e-5 with warmup, fp16 on CUDA, EER as the model-selection
metric. `--freeze-layers 12` fits a smaller GPU; do not raise the learning rate to
achieve the same thing.

## Benchmark

```bash
python -m src.benchmark --manifest data/eval.jsonl --out results/
```

Produces `matrix.csv`: EER, false-positive rate at fixed true-positive rate, and
alert precision at the 0.17 percent banking base rate, for every codec and packet
loss combination, broken down by language.

## Tests

```bash
pytest tests/ -v
```

The important one is `test_label_convention` — it asserts real is 0 and fake is 1,
which is the bug that silently inverted every label in v0.3.

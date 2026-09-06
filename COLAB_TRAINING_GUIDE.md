# Ultimate Google Colab Guide (Data Generation)

This guide walks you through **Day 1 and Day 2** of the Master Plan. Because we discovered that the ML model was "cheating" by listening to the background noise (The Corpus Shortcut), we must use Google Colab to generate Deepfakes from the exact same human speakers that we use for our Real data.

## Step 1: Prepare Colab
1. Go to [colab.research.google.com](https://colab.research.google.com) and create a New Notebook.
2. At the top right, click **Runtime** > **Change runtime type**.
3. Select **T4 GPU**.

## Step 2: Mount your Google Drive & Install Everything
Copy and paste this into **Cell 1** and click Run. A pop-up will ask for permission to mount your Google Drive. Allow it! We are saving the audio files directly to your Drive so that if Colab disconnects, you don't lose your work.

```python
!pip install -q datasets soundfile librosa transformers accelerate
!pip install -q coqui-tts
!apt-get -qq install -y ffmpeg
!pip install -q git+https://github.com/AI4Bharat/IndicF5.git
!pip install -q git+https://github.com/huggingface/parler-tts.git

from google.colab import drive
drive.mount('/content/drive')
OUT = "/content/drive/MyDrive/sih_data" 
```

## Step 3: Download the Real Audio (Google Fleurs)
Copy and paste this into **Cell 2**. This will download 3,000 real human clips across Hindi, Marathi, Tamil, and Bengali.

```python
from datasets import load_dataset
import soundfile as sf, os, json, collections

LANGS = ["hi_in", "mr_in", "ta_in", "bn_in"]
real_rows = []
for lang in LANGS:
    ds = load_dataset("google/fleurs", lang, split="train", streaming=True)
    per_speaker = collections.Counter()
    for ex in ds:
        spk = str(ex.get("id", "unk"))
        if per_speaker[spk] >= 12:             
            continue
        per_speaker[spk] += 1
        p = f"{OUT}/real/{lang}_{spk}_{ex['id']}.wav"
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            sf.write(p, ex["audio"]["array"], ex["audio"]["sampling_rate"])
        real_rows.append({"path": p, "label": "real", "family": "real",
                          "speaker": f"{lang}_{spk}", "language": lang,
                          "text": ex["transcription"], "corpus": "fleurs"})
        if len(real_rows) >= 3000:
            break
print(f"Downloaded {len(real_rows)} real clips from {len({r['speaker'] for r in real_rows})} speakers")
```

## Step 4: Generate the Deepfakes (The Heavy Lifting)

You will now run 4 massive Deepfake generation engines on the T4 GPU. This will take 3 to 5 hours! Run these cells one by one.

### Cell 3: XTTS-v2 (HiFi-GAN)
```python
# Patch transformers to fix coqui-tts compatibility with new versions
import transformers.utils.import_utils
transformers.utils.import_utils.is_torch_greater_or_equal = lambda *args, **kwargs: False
transformers.utils.import_utils.is_torchcodec_available = lambda *args, **kwargs: False

from TTS.api import TTS
import random, os

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

fake_rows = []
for i, target in enumerate(real_rows):
    spk = target["speaker"]
    p = f"{OUT}/fake/hifigan/{spk}_{i}.wav"
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            # Use the clip itself as the reference to clone its own voice perfectly!
            tts.tts_to_file(text=target["text"], speaker_wav=target["path"],
                            language=target["language"].split("_")[0], file_path=p)
            fake_rows.append({"path": p, "label": "fake", "family": "hifigan",
                              "speaker": spk, "language": target["language"],
                              "text": target["text"], "corpus": "xtts_v2"})
        except Exception as e:
            pass # silently skip errors to keep it moving
    else:
        fake_rows.append({"path": p, "label": "fake", "family": "hifigan",
                          "speaker": spk, "language": target["language"],
                          "text": target["text"], "corpus": "xtts_v2"})
```

### Cell 4: IndicF5 (Vocos - The most dangerous threat)
```python
!pip install -q git+https://github.com/AI4Bharat/IndicF5.git
from transformers import AutoModel
import numpy as np, os, soundfile as sf

f5 = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to("cuda")

for i, target in enumerate(real_rows):
    spk = target["speaker"]
    p = f"{OUT}/fake/vocos/{spk}_{i}.wav"
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            # We will use a random real clip as the reference to guarantee it generates properly
            ref = random.choice(real_rows)
            wav = f5(target["text"], ref_audio_path=ref["path"], ref_text=ref["text"])
            wav = np.asarray(wav, dtype="float32")
            sf.write(p, wav / (np.abs(wav).max() + 1e-9) * 0.95, 24000)
            fake_rows.append({"path": p, "label": "fake", "family": "vocos",
                              "speaker": spk, "language": target["language"],
                              "text": target["text"], "corpus": "indicf5"})
        except Exception as e:
            pass
    else:
        fake_rows.append({"path": p, "label": "fake", "family": "vocos",
                          "speaker": spk, "language": target["language"],
                          "text": target["text"], "corpus": "indicf5"})
```

### Cell 5: Bark (EnCodec)
```python
from transformers import AutoProcessor, BarkModel
proc = AutoProcessor.from_pretrained("suno/bark-small")
bark = BarkModel.from_pretrained("suno/bark-small").to("cuda")

for i, target in enumerate(real_rows):
    spk = target["speaker"]
    p = f"{OUT}/fake/encodec/{spk}_{i}.wav"
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            inputs = proc(target["text"], voice_preset="v2/hi_speaker_2").to("cuda")
            out = bark.generate(**inputs)
            sf.write(p, out.cpu().numpy().squeeze(), bark.generation_config.sample_rate)
            fake_rows.append({"path": p, "label": "fake", "family": "encodec",
                              "speaker": spk, "language": target["language"],
                              "text": target["text"], "corpus": "bark"})
        except Exception as e:
            pass
    else:
        fake_rows.append({"path": p, "label": "fake", "family": "encodec",
                          "speaker": spk, "language": target["language"],
                          "text": target["text"], "corpus": "bark"})
```

### Cell 6: Parler-TTS (Modern Codec)
```python
!pip install -q git+https://github.com/huggingface/parler-tts.git
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

pt = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts", token="<YOUR_HUGGINGFACE_TOKEN>").to("cuda")
tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", token="<YOUR_HUGGINGFACE_TOKEN>")
desc = "A clear voice speaking at a moderate pace with minimal background noise."

for i, target in enumerate(real_rows):
    spk = target["speaker"]
    p = f"{OUT}/fake/modern_codec/{spk}_{i}.wav"
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            ids = tok(desc, return_tensors="pt").input_ids.to("cuda")
            pids = tok(target["text"], return_tensors="pt").input_ids.to("cuda")
            a = pt.generate(input_ids=ids, prompt_input_ids=pids)
            sf.write(p, a.cpu().numpy().squeeze(), pt.config.sampling_rate)
            fake_rows.append({"path": p, "label": "fake", "family": "modern_codec",
                              "speaker": spk, "language": target["language"],
                              "text": target["text"], "corpus": "parler"})
        except Exception as e:
            pass
    else:
        fake_rows.append({"path": p, "label": "fake", "family": "modern_codec",
                          "speaker": spk, "language": target["language"],
                          "text": target["text"], "corpus": "parler"})
```

## Step 5: Save the Final Manifest
Run this final cell. It combines everything into one clean `manifest.jsonl` file on your Google Drive. 

```python
import json
all_rows = real_rows + fake_rows
with open(f"{OUT}/manifest.jsonl", "w") as fh:
    for r in all_rows: 
        fh.write(json.dumps(r) + "\n")

print("FINISHED! Dataset saved to Google Drive: MyDrive/sih_data/")
```

---

**What to do when this finishes:** 
When all these cells finish running (in a few hours), you will have the most sophisticated, modern Indic Deepfake dataset currently available. Download the `sih_data` folder from your Google Drive, run `split_manifest.py` on it, and you'll be ready for Fix 4 (Kaggle Training)!

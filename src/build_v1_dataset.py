"""v1.0 Hybrid Dataset Builder (Pre-generated Pipeline).

This script pulls Real audio from Mozilla Common Voice and Fake audio from WaveFake,
maps them to the new 7-way Family taxonomy, aggressively degrades them via the 
telephony pipeline (including SNR noise), and emits a structured JSONL manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path

# Add project root to path so we can import 'src'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import datasets
import edge_tts
import numpy as np
import soundfile as sf

from src import audio_prep, families
from src.telephony_degrader import Condition, degrade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vfd.data")

# WaveFake Vocoders to our 7-way Tier 2 Taxonomy
WAVEFAKE_MAPPING = {
    "WF1": families.LABEL2ID["hifigan"],     # Multi-band MelGAN
    "WF2": families.LABEL2ID["legacy_ar"],   # MelGAN
    "WF3": families.LABEL2ID["legacy_ar"],   # Parallel WaveGAN
    "WF4": families.LABEL2ID["hifigan"],     # HiFi-GAN
    "WF5": families.LABEL2ID["legacy_ar"],   # WaveGlow
    "WF6": families.LABEL2ID["legacy_ar"],   # Full-band MelGAN
}

def get_random_condition(rng: np.random.Generator) -> Condition:
    """Returns a randomized telephony degradation condition."""
    snr = rng.uniform(10, 20) if rng.random() < 0.5 else None
    
    # 9-codec random routing
    codec = rng.choice(["g711_ulaw", "g711_alaw", "opus_nb", "amr_nb", "g722", "none"])
    bitrate = None
    if codec == "opus_nb": 
        bitrate = 16.0
    elif codec == "amr_nb": 
        bitrate = 12.2
        
    pl = 0.03 if rng.random() < 0.5 else 0.0
    return Condition(codec=str(codec), bitrate_kbps=bitrate, snr_db=snr, packet_loss=pl)

def build_v1_dataset(out_dir: str, limit_real: int = 1000, limit_fake: int = 1000) -> None:
    out = Path(out_dir)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "train.jsonl"
    
    # We must seed explicitly to ensure reproducibility across runs
    rng = np.random.default_rng(42)
    records = []
    
    # --- PHASE 1a: REAL AUDIO (Fleurs English) ---
    log.info("Phase 1a: Fetching Real Audio from Google Fleurs (English)")
    try:
        cv_en = datasets.load_dataset("google/fleurs", "en_us", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        cv_en = datasets.load_dataset("google/fleurs", "en_us", split="train", streaming=True)
        
    limit_en = limit_real // 2
    for i, item in enumerate(cv_en):
        if i >= limit_en:
            break
        
        arr = np.asarray(item["audio"]["array"], dtype=np.float32)
        sr = int(item["audio"]["sampling_rate"])
        x = audio_prep.to_mono(arr)
        x = audio_prep.resample(x, sr, audio_prep.TARGET_SR)
        x = audio_prep.trim_silence(x)
        
        cond = get_random_condition(rng)
        y, out_sr = degrade(x, audio_prep.TARGET_SR, cond, rng=rng)
        
        speaker = item.get("client_id", f"fleurs_en_{i}")[:16]
        name = f"train_real_{i:06d}.wav"
        dest = audio_dir / name
        sf.write(dest, y, out_sr, format="WAV", subtype="PCM_16")
        
        records.append({
            "path": str(dest),
            "label": "real",
            "family": families.REAL.idx,
            "speaker": speaker,
            "language": "en",
            "condition": cond.label()
        })
        if (i + 1) % 100 == 0:
            log.info(f"Processed {i + 1} English real audio files")
            
    # Free English Parquet buffers from RAM before pulling Hindi
    del cv_en
    import gc
    gc.collect()

    hindi_transcripts = []

    # --- PHASE 1b: REAL AUDIO (Fleurs Hindi) ---
    log.info("Phase 1b: Fetching Real Audio from Google Fleurs (Hindi)")
    try:
        cv_hi = datasets.load_dataset("google/fleurs", "hi_in", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        cv_hi = datasets.load_dataset("google/fleurs", "hi_in", split="train", streaming=True)
        
    limit_hi = limit_real - limit_en
    for i, item in enumerate(cv_hi):
        if i >= limit_hi:
            break
            
        hindi_transcripts.append(item.get("raw_transcription", "नमस्ते"))
        
        arr = np.asarray(item["audio"]["array"], dtype=np.float32)
        sr = int(item["audio"]["sampling_rate"])
        x = audio_prep.to_mono(arr)
        x = audio_prep.resample(x, sr, audio_prep.TARGET_SR)
        x = audio_prep.trim_silence(x)
        
        cond = get_random_condition(rng)
        y, out_sr = degrade(x, audio_prep.TARGET_SR, cond, rng=rng)
        
        idx = limit_en + i
        speaker = item.get("client_id", f"fleurs_hi_{idx}")[:16]
        name = f"train_real_{idx:06d}.wav"
        dest = audio_dir / name
        sf.write(dest, y, out_sr, format="WAV", subtype="PCM_16")
        
        records.append({
            "path": str(dest),
            "label": "real",
            "family": families.REAL.idx,
            "speaker": speaker,
            "language": "hi",
            "condition": cond.label()
        })
        if (i + 1) % 100 == 0:
            log.info(f"Processed {i + 1} Hindi real audio files")
            
    del cv_hi
    gc.collect()
            
    # --- PHASE 2: FAKE AUDIO (WaveFake English) ---
    log.info("Phase 2: Fetching Fake Audio from WaveFake (English)")
    wf = datasets.load_dataset("ajaykarthick/wavefake-audio", split="train", streaming=True, trust_remote_code=True)
    
    limit_fake_en = limit_fake // 2
    for i, item in enumerate(wf):
        if i >= limit_fake_en:
            break
            
        arr = np.asarray(item["audio"]["array"], dtype=np.float32)
        sr = int(item["audio"]["sampling_rate"])
        x = audio_prep.to_mono(arr)
        x = audio_prep.resample(x, sr, audio_prep.TARGET_SR)
        x = audio_prep.trim_silence(x)
        
        cond = get_random_condition(rng)
        y, out_sr = degrade(x, audio_prep.TARGET_SR, cond, rng=rng)
        
        vocoder_id = item.get("real_or_fake", "WF2")
        
        # WaveFake includes genuine LJSpeech files marked as "real"
        if vocoder_id == "real":
            continue
            
        family_idx = WAVEFAKE_MAPPING.get(vocoder_id, families.LABEL2ID["legacy_ar"])
        
        name = f"train_fake_{i:06d}.wav"
        dest = audio_dir / name
        sf.write(dest, y, out_sr, format="WAV", subtype="PCM_16")
        
        records.append({
            "path": str(dest),
            "label": "fake",
            "family": family_idx,
            "speaker": f"wf_{vocoder_id}",
            "language": "en",
            "condition": cond.label()
        })
        if (i + 1) % 100 == 0:
            log.info(f"Processed {i + 1} fake audio files")
            
    del wf
    gc.collect()

    # --- PHASE 3: FAKE AUDIO (Dynamic Edge-TTS Hindi) ---
    log.info("Phase 3: Synthesizing Fake Audio via Edge-TTS (Hindi)")
    
    async def generate_hindi_fakes():
        import librosa
        voices = ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]
        for i, text in enumerate(hindi_transcripts):
            if i >= (limit_fake - limit_fake_en):
                break
                
            idx = limit_fake_en + i
            dest_mp3 = audio_dir / f"temp_{idx}.mp3"
            dest_wav = audio_dir / f"train_fake_{idx:06d}.wav"
            voice = rng.choice(voices)
            
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(dest_mp3))
                
                # Jitter for rate limits
                await asyncio.sleep(0.5 + rng.random())
                
                # Load MP3 natively resampled to 16kHz
                arr, sr = librosa.load(str(dest_mp3), sr=audio_prep.TARGET_SR)
                
                x = audio_prep.to_mono(arr)
                x = audio_prep.trim_silence(x)
                
                cond = get_random_condition(rng)
                y, out_sr = degrade(x, audio_prep.TARGET_SR, cond, rng=rng)
                
                sf.write(dest_wav, y, out_sr, format="WAV", subtype="PCM_16")
                dest_mp3.unlink(missing_ok=True)
                
                records.append({
                    "path": str(dest_wav),
                    "label": "fake",
                    "family": families.LABEL2ID["hifigan"],
                    "speaker": f"edge_{voice}",
                    "language": "hi",
                    "condition": cond.label()
                })
                
                if (i + 1) % 50 == 0:
                    log.info(f"Processed {i + 1} Edge-TTS Hindi files")
                    gc.collect()
            except Exception as e:
                log.warning(f"TTS synthesis failed for clip {idx}: {e}")
                
    asyncio.run(generate_hindi_fakes())
    gc.collect()

    # --- PHASE 4: MANIFEST EXPORT ---
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    log.info(f"Done! Manifest saved to {manifest_path} with {len(records)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the v1.0 hybrid dataset")
    parser.add_argument("--out", default="./data")
    parser.add_argument("--real", type=int, default=1000)
    parser.add_argument("--fake", type=int, default=1000)
    args = parser.parse_args()
    
    build_v1_dataset(args.out, limit_real=args.real, limit_fake=args.fake)

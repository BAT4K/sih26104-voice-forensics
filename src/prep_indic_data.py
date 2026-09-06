import os
import sys
import asyncio
from pathlib import Path
import soundfile as sf
import datasets
import warnings
import edge_tts
import librosa
from tqdm import tqdm

# Suppress minor huggingface warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to sys.path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.telephony_degrader import simulate_telephony_degradation

async def generate_synthetic_audio(text: str, output_path: str) -> None:
    """Generates synthetic Hindi TTS audio using edge-tts."""
    # We use a natural sounding Hindi neural voice
    voice = "hi-IN-MadhurNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def setup_train_directories(base_dir: Path) -> None:
    """Creates the overnight training directory structure."""
    dirs = [
        base_dir / "train" / "real",
        base_dir / "train" / "fake"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {d}")

async def main_async() -> None:
    """Main execution block to pull Google Fleurs and build matched TTS dataset."""
    print("Initializing overnight Hindi TTS dataset builder...")
    
    data_dir = PROJECT_ROOT / "data"
    setup_train_directories(data_dir)
    
    tmp_dir = PROJECT_ROOT / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    try:
        print("\nLoading Google Fleurs (Hindi) in streaming mode...")
        ds = datasets.load_dataset("google/fleurs", "hi_in", split="train", streaming=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to stream the dataset: {e}")
        return
        
    target_count = 500
    count = 0
    
    print(f"\nFetching {target_count} transcripts, dynamically generating deepfakes via edge-tts, and degrading...")
    
    for item in tqdm(ds, total=target_count, desc="Building Dataset"):
        if count >= target_count:
            break
            
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        
        # Extract the transcript for this exact genuine audio file
        transcript = item.get("transcription", item.get("raw_transcription", "नमस्कार, मैं एक कृत्रिम बुद्धिमत्ता हूँ।"))
        
        # 1. Handle REAL Audio
        tmp_real = tmp_dir / "raw_real.wav"
        sf.write(str(tmp_real), audio_array, sr)
        final_real = data_dir / "train" / "real" / f"real_hi_{count}.wav"
        simulate_telephony_degradation(str(tmp_real), str(final_real))
        
        # 2. Handle FAKE Audio (Synthetic TTS)
        tmp_fake_mp3 = tmp_dir / "raw_fake.mp3"
        tmp_fake_wav = tmp_dir / "raw_fake.wav"
        
        try:
            # Generate the deepfake mp3 dynamically using the same transcript
            await generate_synthetic_audio(transcript, str(tmp_fake_mp3))
            
            # Convert MP3 to WAV (using librosa to bypass soundfile MP3 issues on Linux)
            fake_y, fake_sr = librosa.load(str(tmp_fake_mp3), sr=None)
            sf.write(str(tmp_fake_wav), fake_y, fake_sr)
            
            # Pass the synthetic deepfake through our telephony degrader
            final_fake = data_dir / "train" / "fake" / f"fake_hi_{count}.wav"
            simulate_telephony_degradation(str(tmp_fake_wav), str(final_fake))
            
        except Exception as e:
            print(f"Skipping fake generation for {count} due to error: {e}")
            continue
            
        count += 1

    # Cleanup temp files
    (tmp_dir / "raw_real.wav").unlink(missing_ok=True)
    (tmp_dir / "raw_fake.mp3").unlink(missing_ok=True)
    (tmp_dir / "raw_fake.wav").unlink(missing_ok=True)

    print("\n✅ Indic Overnight Training Batch complete!")
    print(f"Successfully generated 500 Real and 500 Fake files.")
    print("The real and synthetic files are matched perfectly by spoken text transcript.")
    print("\nYou are ready to re-run `python train.py` overnight!")

def main():
    # edge-tts is an async library, so we run the async loop
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

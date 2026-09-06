import json
import random
import os
from collections import defaultdict

def create_strict_splits(manifest_path, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Creates strict speaker-disjoint splits to prevent data leakage.
    Ensures that a speaker seen in training NEVER appears in validation or testing.
    """
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Ensure you downloaded it from Google Drive!")
        return

    # 1. Read all rows
    with open(manifest_path, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} total audio files from {manifest_path}")

    # 2. Group data by speaker to prevent leakage
    # We use speaker as the key so that all of one person's audio stays together
    speaker_groups = defaultdict(list)
    for row in data:
        # If speaker is missing, group them in a generic 'unknown' bucket 
        # (though all our data should have speaker tags)
        speaker = row.get("speaker", "unknown_speaker")
        speaker_groups[speaker].append(row)

    unique_speakers = list(speaker_groups.keys())
    print(f"Found {len(unique_speakers)} unique speakers.")

    # 3. Shuffle speakers randomly (with a fixed seed for reproducibility)
    random.seed(42)
    random.shuffle(unique_speakers)

    # 4. Calculate split indices based on speaker count
    num_speakers = len(unique_speakers)
    train_end = int(num_speakers * train_ratio)
    val_end = train_end + int(num_speakers * val_ratio)

    train_speakers = set(unique_speakers[:train_end])
    val_speakers = set(unique_speakers[train_end:val_end])
    test_speakers = set(unique_speakers[val_end:])

    # 5. Distribute the actual audio files based on which group their speaker landed in
    train_data = []
    val_data = []
    test_data = []

    for spk, rows in speaker_groups.items():
        if spk in train_speakers:
            train_data.extend(rows)
        elif spk in val_speakers:
            val_data.extend(rows)
        else:
            test_data.extend(rows)

    # 6. Create the separate output folder
    os.makedirs(output_dir, exist_ok=True)

    # 7. Write the split files
    splits = {
        "train.jsonl": train_data,
        "val.jsonl": val_data,
        "test.jsonl": test_data
    }

    for filename, split_data in splits.items():
        # Shuffle the individual rows within the split so it's not all one speaker in a row
        random.shuffle(split_data)
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'w') as f:
            for row in split_data:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(split_data)} files to {out_path}")

    print("\n✅ Strict speaker-disjoint splitting complete!")
    print(f"The model will train on {len(train_speakers)} speakers, and be tested on {len(test_speakers)} completely unseen speakers.")

if __name__ == "__main__":
    # Assumes the downloaded manifest is placed in the project root
    INPUT_MANIFEST = "manifest.jsonl"
    OUTPUT_FOLDER = "dataset_splits"
    
    create_strict_splits(INPUT_MANIFEST, OUTPUT_FOLDER)

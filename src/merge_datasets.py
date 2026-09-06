import json
import os

def merge_datasets():
    all_data = []

    # 1. Load the original data (data/train.jsonl, data/eval.jsonl)
    for old_file in ['data/train.jsonl', 'data/eval.jsonl']:
        if os.path.exists(old_file):
            with open(old_file, 'r') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))
            print(f"Loaded {old_file}")

    # 2. Load the new Colab data and fix the paths
    sih_manifest = 'sih_data/manifest.jsonl'
    if os.path.exists(sih_manifest):
        with open(sih_manifest, 'r') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    # Fix path: /content/drive/MyDrive/sih_data/... -> sih_data/...
                    old_path = row['path']
                    if "/content/drive/MyDrive/sih_data/" in old_path:
                        row['path'] = old_path.replace("/content/drive/MyDrive/sih_data/", "sih_data/")
                    all_data.append(row)
        print(f"Loaded {sih_manifest} and fixed paths")

    # 3. Save to a master manifest
    with open('manifest.jsonl', 'w') as f:
        for row in all_data:
            f.write(json.dumps(row) + '\n')
            
    print(f"\n✅ Merged EVERYTHING into manifest.jsonl!")
    print(f"Total Combined Files: {len(all_data)}")

if __name__ == "__main__":
    merge_datasets()

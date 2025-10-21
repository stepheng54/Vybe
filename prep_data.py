import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from features.extract_features import extract_features
from models.similarity_search import SimilaritySearch

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
BROKEN_DIR = "data/broken_raw"
META_FILE = os.path.join(PROCESSED_DIR, "metadata.csv")
INDEX_FILE = "models/faiss_index.bin"

# Move a corrupted or unreadable file to data/broken_raw/, preserving subfolders
def safe_move_to_broken(src_path):
    rel_path = os.path.relpath(src_path, RAW_DIR)
    dest_path = os.path.join(BROKEN_DIR, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        shutil.move(src_path, dest_path)
        print(f" Moved to {dest_path}")
    except Exception as e:
        print(f" Could not move file ({e})")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    os.makedirs(BROKEN_DIR, exist_ok=True)

    print(f"Checking {RAW_DIR} for new audio files...")
    songs = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith((".mp3", ".wav", ".flac")):
                songs.append(os.path.join(root, f))

    if not songs:
        print("No audio files found in data/raw/. Add songs first.")
        return

    metadata = []

    # Temporary placeholder; will update later with actual feature length
    search_model = SimilaritySearch(feature_dim=51)

    for song_path in tqdm(songs, desc="Extracting features"):
        file_name = os.path.basename(song_path)
        feature_path = os.path.join(
            PROCESSED_DIR,
            file_name.replace(".mp3", ".npy").replace(".wav", ".npy").replace(".flac", ".npy"),
        )

        # Skip if already processed
        if os.path.exists(feature_path):
            try:
                vec = np.load(feature_path)
            except Exception as e:
                print(f"[ERROR] Corrupted feature file {feature_path}: {e}")
                os.remove(feature_path)
                continue
        else:
            try:
                vec = extract_features(song_path)
            except Exception as e:
                print(f"[ERROR] Failed to process {song_path}: {e}")
                safe_move_to_broken(song_path)
                continue

            if vec.size > 0:
                np.save(feature_path, vec)
            else:
                print(f"[WARN] Empty features — skipping {file_name}")
                safe_move_to_broken(song_path)
                continue

        # Update FAISS feature dimension dynamically if first valid vector
        if len(metadata) == 0:
            search_model = SimilaritySearch(feature_dim=len(vec))

        # Add to FAISS index
        search_model.add_song(file_name, vec)

        # Add metadata entry
        metadata.append({
            "filename": file_name,
            "feature_path": os.path.basename(feature_path),
            "length": len(vec)
        })

    # Save metadata
    pd.DataFrame(metadata).to_csv(META_FILE, index=False)
    print(f"\n Saved metadata to {META_FILE}")

    # Save FAISS index
    search_model.save(INDEX_FILE)
    print(f"Saved FAISS index to {INDEX_FILE}")

    print("\n Dataset complete and ready for use.")


if __name__ == "__main__":
    main()

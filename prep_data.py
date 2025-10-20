import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from features.extract_features import extract_features
from models.similarity_search import SimilaritySearch

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
META_FILE = os.path.join(PROCESSED_DIR, "metadata.csv")
INDEX_FILE = "models/faiss_index.bin"


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

    print(f"Checking {RAW_DIR} for new audio files")
    songs = [
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".mp3", ".wav", ".flac"))
    ]

    if not songs:
        print("No audio files found in data/raw/. Add songs first.")
        return

    metadata = []
    search_model = SimilaritySearch(feature_dim=51)

    for song_path in tqdm(songs, desc="Extracting features"):
        file_name = os.path.basename(song_path)
        feature_path = os.path.join(PROCESSED_DIR, file_name.replace(".mp3", ".npy").replace(".wav", ".npy"))

        # Skip if already processed
        if os.path.exists(feature_path):
            vec = np.load(feature_path)
        else:
            vec = extract_features(song_path)
            if vec.size > 0:
                np.save(feature_path, vec)
            else:
                print(f"Skipping {file_name}")
                continue

        # Add to FAISS index
        search_model.add_song(file_name, vec)

        # Save metadata
        metadata.append({"filename": file_name, "feature_path": os.path.basename(feature_path)})

    # Save metadata CSV
    pd.DataFrame(metadata).to_csv(META_FILE, index=False)
    print(f"Saved metadata to {META_FILE}")

    # Save FAISS index
    search_model.save(INDEX_FILE)
    print(f"Saved FAISS index to {INDEX_FILE}")

    print("\n Dataset complete and ready for use")


if __name__ == "__main__":
    main()

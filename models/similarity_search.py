"""
similarity_search.py
--------------------
Builds and queries a similarity index for songs based on extracted audio features.
Uses FAISS for efficient nearest-neighbor search.
"""

import numpy as np
import faiss
import os

class SimilaritySearch:

    # Initializes a FAISS index for L2 (Euclidean) distance.
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.index = faiss.IndexFlatL2(feature_dim)
        self.song_ids = []  # keep track of filenames or IDs

    # Adds a new song feature vector to the index.
    def add_song(self, song_id: str, feature_vector: np.ndarray):
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        self.index.add(feature_vector.astype("float32"))
        self.song_ids.append(song_id)

    # Searches for the k most similar songs to the given feature vector.
    def search(self, query_vector: np.ndarray, k: int = 5):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector.astype("float32"), k)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.song_ids):
                results.append((self.song_ids[i], dist))
        return results

    # Saves the FAISS index and song metadata to disk.
    def save(self, path: str = "models/faiss_index.bin"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        np.save(path.replace(".bin", "_ids.npy"), np.array(self.song_ids))

    # Loads a FAISS index and song metadata from disk.
    def load(self, path: str = "models/faiss_index.bin"):
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            self.song_ids = np.load(path.replace(".bin", "_ids.npy")).tolist()
        else:
            raise FileNotFoundError(f"No FAISS index found at {path}")


if __name__ == "__main__":
    # Example usage (test run)
    from features.extract_features import extract_features

    # 
    songs = []

    model = SimilaritySearch(feature_dim=51)

    for song in songs:
        vec = extract_features(song)
        if vec.size > 0:
            model.add_song(song, vec)

    query = extract_features("../data/raw/song1.mp3")
    results = model.search(query, k=3)
    print("\nSimilar Songs:")
    for song_id, dist in results:
        print(f"{song_id} (distance={dist:.3f})")
from models.similarity_search import SimilaritySearch
from features.extract_features import extract_features
import numpy as np
import pandas as pd
import librosa
import soundfile as sf


# Load FAISS index + metadata
model = SimilaritySearch(feature_dim=64)
model.load("models/faiss_index.bin")

meta = pd.read_csv("data/processed/metadata.csv")

# Pick a NEW song not in the dataset
query_song = "C:\\Users\\sgilt\\OneDrive\\Desktop\\Vybe\\South Arcade - 2005 (Official Video).mp3"

# Load a 20-second clip (starting at a random offset if the song is longer)
duration = 20  # seconds
total_duration = librosa.get_duration(path=query_song)
offset = 0 if total_duration <= duration else np.random.uniform(0, total_duration - duration)

print(f"Extracting {duration}s clip starting at {offset:.2f}s")

# Extract features using only that clip
y, sr = librosa.load(query_song, sr=None, mono=True, offset=offset, duration=duration)
clip_path = "temp_clip.wav"
sf.write(clip_path, y, sr)

query_vec = extract_features(clip_path)
print("Query vector shape:", query_vec.shape)

# Search top 5 similar songs
results = model.search(query_vec, k=5)

print("\nTop similar songs:")
for idx, (song, dist) in enumerate(results, start=1):
    print(f"{idx}. {song} (distance: {dist:.4f})")

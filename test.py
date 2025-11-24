import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import os
from models.similarity_search import SimilaritySearch
from features.extract_features import extract_features

QUERY_SONG = "C:\\Users\\sgilt\\OneDrive\\Desktop\\Vybe\\South Arcade - 2005.mp3"
CLIP_DURATION = 30
INDEX_PATH = "models/faiss_index.bin"
META_PATH = "data/processed/metadata.csv"

model = SimilaritySearch(feature_dim=64)
model.load(INDEX_PATH)
meta = pd.read_csv(META_PATH)
y, sr = librosa.load(QUERY_SONG, sr=None, mono=True)

# This part finds the most energetic segment in the song, or the part of the song
# that is most reflective of the song as a whole
hop_length = 512
frame_length = 2048
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
frames = np.arange(len(rms))
times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
window = int((CLIP_DURATION * sr) / hop_length)
if window < len(rms):
    avg_rms = np.convolve(rms, np.ones(window), "valid") / window
    best_frame = np.argmax(avg_rms)
    offset = float(times[best_frame])
else:
    offset = 0.0

# Extract just that portion for searching
y_clip, sr = librosa.load(QUERY_SONG, sr=sr, mono=True, offset=offset, duration=CLIP_DURATION)
clip_path = "temp_clip.wav"
sf.write(clip_path, y_clip, sr)

query_vec = extract_features(clip_path)

results = model.search(query_vec, k=5)

print("\nTop similar songs:")
for i, (song, dist) in enumerate(results, start=1):
    print(f"{i}. {song} (similarity: {dist:.4f})")

# Don't need to store the clip permanently
if os.path.exists(clip_path):
    os.remove(clip_path)
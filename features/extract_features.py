"""
extract_features.py
-------------------
Extracts core audio features (tempo, MFCCs, chroma) from an audio file.
These features represent each track numerically for use in similarity search.
"""

import librosa
import numpy as np

def extract_features(file_path: str) -> np.ndarray:
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # --- Feature Extraction ---
        # 1. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # 2. MFCCs (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 3. Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # --- Combine into one vector ---
        feature_vector = np.hstack([
            np.array([tempo]),
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std
        ])

        return feature_vector

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return np.array([])


if __name__ == "__main__":
    # Example usage
    test_file = "../data/raw/sample.mp3"  # Change this path as needed
    features = extract_features(test_file)
    print("Extracted feature vector shape:", features.shape)

import librosa
import numpy as np

def flatten(x) -> np.ndarray:
    if x is None or not hasattr(x, "size") or x.size == 0:
        return np.zeros(1)
    return np.ravel(x)  # guarantees 1D

def extract_features(file_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        if y is None or len(y) < sr / 2:  # skip very short or empty clips
            print(f"[WARN] {file_path} too short or unreadable, skipping.")
            return np.array([])

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = np.array([tempo]) 

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        # Combine features
        parts = [
            flatten(tempo),
            flatten(mfcc_mean), flatten(mfcc_std),
            flatten(chroma_mean), flatten(chroma_std),
            flatten(contrast_mean), flatten(tonnetz_mean)
        ]
        feature_vector = np.concatenate(parts).astype(np.float32)

        return feature_vector

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return np.array([])


if __name__ == "__main__":
    # Test the feature extraction
    sample_file = "data/raw/fma_small/000/000002.mp3"
    features = extract_features(sample_file)
    print(f"Extracted feature vector shape: {features.shape}")

import librosa
import numpy as np

def extract_features(file_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # MFCCs (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Combine into one vector
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
    # Test the feature extraction
    sample_file = ""
    features = extract_features(sample_file)
    print(f"Extracted features from {sample_file}: {features}")
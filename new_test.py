import os
import sys
import numpy as np
import pandas as pd
import faiss, joblib
import librosa
import soundfile as sf

from features.extract_features import extract_features

PROCESSED_DIR = "data/processed"
INDEX_FILE = "models/faiss_index.bin"
SCALER_FILE = "models/feature_scaler.pkl"
MAPPING_FILE = os.path.join(PROCESSED_DIR, "index_mapping.csv")
LIBRARY_FILE = os.path.join(PROCESSED_DIR, "library.csv")

CLIP_DURATION = 30.0  # seconds for the smart clip
TEMP_CLIP_PATH = "temp_query_clip.wav"

# find song name and track id based on filename
def lookup_track_by_filename(filename: str, lib: pd.DataFrame):
    row = lib.loc[lib["filename"] == filename]
    if row.empty:
        return None, filename
    row = row.iloc[0]
    tid = None
    if "track_id" in row and not pd.isna(row["track_id"]):
        tid = int(row["track_id"])
    return tid, row["display"]

CHERRY_FILENAME = "South Arcade - FEAR OF HEIGHTS.mp3"

# for testing: get the vector for a known cherry-picked track
# to verify that certain songs are found as similar or not
def get_cherry_vector_by_filename(index, mapping: pd.DataFrame, lib: pd.DataFrame):
    # Find mapping row for the filename
    row_map = mapping.loc[mapping["filename"] == CHERRY_FILENAME]
    if row_map.empty:
        return None, ""

    row_map = row_map.iloc[0]
    idx_pos = int(row_map["index_pos"])

    # Reconstruct vector from FAISS index 
    cherry_vec = index.reconstruct(idx_pos).reshape(1, -1).astype("float32")

    row_lib = lib.loc[lib["filename"] == CHERRY_FILENAME]
    if row_lib.empty:
        display = CHERRY_FILENAME
    else:
        display = row_lib.iloc[0].get("display", CHERRY_FILENAME)

    return cherry_vec, display


# find the most energetic segment in the song, which is the part 
# of the song that has the most stuff going on
def select_smart_clip(query_path: str, duration: float = CLIP_DURATION) -> str:
    y, sr = librosa.load(query_path, sr=None, mono=True)

    total_len_sec = len(y) / sr
    if total_len_sec <= duration:
        # Song is short so just use the whole thing
        print(f"File shorter than {duration:.1f}s; using entire audio.")
        sf.write(TEMP_CLIP_PATH, y, sr)
        return TEMP_CLIP_PATH

    # Compute RMS energy
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # Number of RMS frames that span the desired duration
    window_frames = int((duration * sr) / hop_length)
    if window_frames <= 0 or window_frames > len(rms):
        # Song is short so just take from middle
        offset = max(0.0, (total_len_sec - duration) / 2.0)
    else:
        # Compute moving average of RMS over the window
        avg_rms = np.convolve(rms, np.ones(window_frames), mode="valid") / window_frames
        best_frame = int(np.argmax(avg_rms))
        offset = float(times[best_frame])

        # Safety clamp so we don't overshoot the end
        if offset + duration > total_len_sec:
            offset = max(0.0, total_len_sec - duration)

    print(f"Selected {duration:.1f}s clip starting at {offset:.2f}s (highest energy region).")

    # Extract that segment and write it to temp WAV
    y_clip, _ = librosa.load(query_path, sr=sr, mono=True, offset=offset, duration=duration)
    sf.write(TEMP_CLIP_PATH, y_clip, sr)
    return TEMP_CLIP_PATH


def main():
    if len(sys.argv) > 1:
        query_path = sys.argv[1]
    else:
        query_path = input("Enter path to audio file (mp3/wav/flac): ").strip()

    if not os.path.exists(query_path):
        print(f"File not found: {query_path}")
        return

    index = faiss.read_index(INDEX_FILE)
    scaler = joblib.load(SCALER_FILE)
    mapping = pd.read_csv(MAPPING_FILE)      
    lib = pd.read_csv(LIBRARY_FILE)          

    temp_clip = select_smart_clip(query_path, duration=CLIP_DURATION)

    q_vec = extract_features(temp_clip)
    if not isinstance(q_vec, np.ndarray) or q_vec.size == 0:
        # Clean up temp file
        if os.path.exists(TEMP_CLIP_PATH):
            os.remove(TEMP_CLIP_PATH)
        return

    q_vec = q_vec.astype("float32").reshape(1, -1)

    # scale + normalize using dataset scaler
    q_vec = scaler.transform(q_vec).astype("float32")
    faiss.normalize_L2(q_vec)

    cherry_vec, cherry_disp = get_cherry_vector_by_filename(index, mapping, lib)
    if cherry_vec is not None:
        sim = float(np.dot(q_vec, cherry_vec.T)[0, 0])
        print(f"\nSimilarity to cherry-picked track:")
        print(f"  {cherry_disp}  (cosine similarity: {sim:.3f})")

    k = 11
    D, I = index.search(q_vec, k)

    print("\nTop similar tracks in your library:")

    shown = 0
    for idx, score in zip(I[0], D[0]):
        hit = mapping.loc[mapping["index_pos"] == idx]
        if hit.empty:
            continue
        hit = hit.iloc[0]
        fname = hit["filename"]
        tid, disp = lookup_track_by_filename(fname, lib)
        print(f"{shown+1}. ID {tid if tid is not None else '??'} | {disp}  (cosine: {score:.3f})")
        shown += 1
        if shown >= 5:
            break
    
    # don't need to keep the temp clip
    if os.path.exists(TEMP_CLIP_PATH):
        os.remove(TEMP_CLIP_PATH)


if __name__ == "__main__":
    main()

import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import faiss, joblib
import librosa
import soundfile as sf
from features.extract_features import extract_features

PROCESSED_DIR = "data/processed"
INDEX_FILE = "models/faiss_index.bin"
SCALER_FILE = "models/feature_scaler.pkl"
MAPPING_FILE = os.path.join(PROCESSED_DIR, "index_mapping.csv")
LIBRARY_FILE = os.path.join(PROCESSED_DIR, "library.csv")

CLIP_DURATION = 30.0


def lookup_track_by_filename(filename: str, lib: pd.DataFrame):
    row = lib.loc[lib["filename"] == filename]
    if row.empty:
        return None, filename
    row = row.iloc[0]
    tid = None
    if "track_id" in row and not pd.isna(row["track_id"]):
        tid = int(row["track_id"])
    return tid, row["display"]


def select_smart_clip_to_path(query_path: str, out_path: str, duration: float = CLIP_DURATION) -> str:
    y, sr = librosa.load(query_path, sr=None, mono=True)

    total_len_sec = len(y) / sr
    if total_len_sec <= duration:
        sf.write(out_path, y, sr)
        return out_path

    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    window_frames = int((duration * sr) / hop_length)
    if window_frames <= 0 or window_frames > len(rms):
        offset = max(0.0, (total_len_sec - duration) / 2.0)
    else:
        avg_rms = np.convolve(rms, np.ones(window_frames), mode="valid") / window_frames
        best_frame = int(np.argmax(avg_rms))
        offset = float(times[best_frame])
        if offset + duration > total_len_sec:
            offset = max(0.0, total_len_sec - duration)

    y_clip, _ = librosa.load(query_path, sr=sr, mono=True, offset=offset, duration=duration)
    sf.write(out_path, y_clip, sr)
    return out_path


@st.cache_resource
def load_assets():
    for path in [INDEX_FILE, SCALER_FILE, MAPPING_FILE, LIBRARY_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    index = faiss.read_index(INDEX_FILE)
    scaler = joblib.load(SCALER_FILE)
    mapping = pd.read_csv(MAPPING_FILE)
    lib = pd.read_csv(LIBRARY_FILE)

    return index, scaler, mapping, lib


def search_similar(query_audio_path: str, k: int = 11, top_n: int = 5):
    index, scaler, mapping, lib = load_assets()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_clip:
        clip_path = tmp_clip.name

    try:
        select_smart_clip_to_path(query_audio_path, clip_path, duration=CLIP_DURATION)

        q_vec = extract_features(clip_path)
        if not isinstance(q_vec, np.ndarray) or q_vec.size == 0:
            return []

        q_vec = q_vec.astype("float32").reshape(1, -1)
        q_vec = scaler.transform(q_vec).astype("float32")
        faiss.normalize_L2(q_vec)

        D, I = index.search(q_vec, k)

        results = []
        shown = 0
        for idx, score in zip(I[0], D[0]):
            hit = mapping.loc[mapping["index_pos"] == idx]
            if hit.empty:
                continue
            fname = hit.iloc[0]["filename"]
            tid, disp = lookup_track_by_filename(fname, lib)
            results.append({
                "rank": shown + 1,
                "track_id": tid,
                "display": disp,
                "similarity": float(score),
                "filename": fname
            })
            shown += 1
            if shown >= top_n:
                break

        return results

    finally:
        if os.path.exists(clip_path):
            os.remove(clip_path)


# Streamlit UI
st.set_page_config(page_title="Vybe", layout="centered")

st.title("Vybe")
st.caption("Upload a song and get the most similar tracks from the library.")

try:
    load_assets()
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac"])
top_n = st.slider("Show top N results", min_value=3, max_value=10, value=5)
k = st.slider("Search breadth (k) -> pick top N results from these", min_value=top_n, max_value=25, value=max(11, top_n))

if uploaded is None:
    st.info("Upload a file to run the demo.")
    st.stop()

suffix = "." + uploaded.name.split(".")[-1].lower()
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
    tmp_in.write(uploaded.getbuffer())
    query_path = tmp_in.name

try:
    with st.spinner("Searching..."):
        results = search_similar(query_path, k=k, top_n=top_n)

    if not results:
        st.error("Could not extract features from this file. Try a different audio file/format.")
        st.stop()

    st.subheader("Top similar songs:")
    for r in results:
        tid = r["track_id"] if r["track_id"] is not None else "Not found"
        st.write(f"**{r['rank']}.** ID {tid} | {r['display']}  \nSimilarity: `{r['similarity']:.3f}`")

finally:
    if os.path.exists(query_path):
        os.remove(query_path)

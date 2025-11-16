import os
import re
import pandas as pd

RAW_DIR = "data/raw/fma_small" 
TRACKS_CSV = "data/external/tracks.csv"
LIB_OUT = "data/processed/library.csv"


def load_tracks() -> pd.DataFrame:
    if not os.path.exists(TRACKS_CSV):
        raise FileNotFoundError(f"tracks.csv not found at: {os.path.abspath(TRACKS_CSV)}")

    # tracks.csv has a 2-row header (album / comments etc.)
    df = pd.read_csv(TRACKS_CSV, header=[0, 1], index_col=0)

    # Flatten MultiIndex columns: ('track','title') -> 'track_title'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{a}_{b}" for a, b in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    df = df.reset_index()
    if "index" in df.columns and "track_id" not in df.columns:
        df = df.rename(columns={"index": "track_id"})

    def col(name: str):
        return df[name] if name in df.columns else pd.NA

    tracks = pd.DataFrame({
    "track_id": df["track_id"].astype(int),
    "title": col("track_title"),
    "artist": col("artist_name"),
    "genre_top": col("track_genre_top"),
    "duration": col("track_duration"),
    })

    return tracks


def scan_files() -> pd.DataFrame:
 rows = []
 for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if not f.lower().endswith(".mp3"):
            continue
        m = re.match(r"(\d{6})\.mp3$", f)
        if not m:
            continue
        tid = int(m.group(1))
        rel = os.path.join(root, f).replace("\\", "/")
        rows.append({"track_id": tid, "filename": f, "rel_path": rel})
 return pd.DataFrame(rows)


def main():
 tracks = load_tracks()
 files = scan_files()

 if files.empty:
    print(f"No MP3 files found under {RAW_DIR}")
    return

 lib = files.merge(tracks, on="track_id", how="left")

 # Fallbacks if metadata missing
 lib["title"] = lib["title"].fillna(lib["filename"])
 lib["artist"] = lib["artist"].fillna("Unknown Artist")
 lib["display"] = lib["title"] + " — " + lib["artist"]

 os.makedirs(os.path.dirname(LIB_OUT), exist_ok=True)
 lib.to_csv(LIB_OUT, index=False)
 print(f"Saved {LIB_OUT} with {len(lib)} rows")


if __name__ == "__main__":
 main()
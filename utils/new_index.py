import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import faiss, joblib

PROCESSED_DIR = "data/processed"
META_FILE = os.path.join(PROCESSED_DIR, "metadata.csv")
INDEX_FILE = "models/faiss_index.bin"
SCALER_FILE = "models/feature_scaler.pkl"
MAPPING_FILE = os.path.join(PROCESSED_DIR, "index_mapping.csv")


def main():
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

    meta = pd.read_csv(META_FILE)

    vectors = []
    kept_rows = []

    for i, row in meta.iterrows():
        fpath = os.path.join(PROCESSED_DIR, row["feature_path"])
        if not os.path.exists(fpath):
            print(f"missing feature file: {fpath}")
            continue
        try:
            vec = np.load(fpath)
        except Exception as e:
            print(f"failed to load {fpath}: {e}")
            continue

        if not isinstance(vec, np.ndarray) or vec.size == 0:
            continue

        vectors.append(vec.astype("float32"))
        kept_rows.append(row)

    if not vectors:
        print("No vectors loaded")
        return

    meta_new = pd.DataFrame(kept_rows).reset_index(drop=True)
    X = np.stack(vectors).astype("float32")

    # Standardize then L2-normalize for cosine similarity 
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    Xz = scaler.transform(X).astype("float32")
    faiss.normalize_L2(Xz)

    index = faiss.IndexFlatIP(Xz.shape[1])
    index.add(Xz)

    faiss.write_index(index, INDEX_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Saved index → {INDEX_FILE}")
    print(f"Saved scaler → {SCALER_FILE}")

    # Save mapping from FAISS index position to filename / feature_path
    meta_new["index_pos"] = meta_new.index
    meta_new.to_csv(META_FILE, index=False)           
    meta_new[["index_pos", "filename", "feature_path"]].to_csv(
        MAPPING_FILE, index=False
    )
    print(f"Saved index mapping → {MAPPING_FILE}")


if __name__ == "__main__":
    main()

from models.similarity_search import SimilaritySearch
import numpy as np
import pandas as pd

# Load index + metadata
model = SimilaritySearch(feature_dim=51)
model.load("models/faiss_index.bin")

meta = pd.read_csv("data/processed/metadata.csv")

# Pick any song
query_path = "data/processed/" + meta.iloc[0, 0].replace(".mp3", ".npy")
print(query_path)
query_vec = np.load(query_path)

# Search top 5 similar songs
results = model.search(query_vec, k=5)
print("Top similar songs:", results)
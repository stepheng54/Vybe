"""
This file was created to confirm that the new feature extraction
actually produces different vectors compared to the old version.
I made this because my manual tests were showing the exact same results
but that was because I was accidentally still using the old feature extraction code.

"""

import numpy as np, os
import random

old_dir = "data/processed_old"
new_dir = "data/processed"

# pick a file that exists in both folders
file = next(f for f in os.listdir(new_dir) if f.endswith(".npy") and os.path.exists(os.path.join(old_dir, f)))

old_vec = np.load(os.path.join(old_dir, file))
new_vec = np.load(os.path.join(new_dir, file))

print("Old vector length:", len(old_vec))
print("New vector length:", len(new_vec))

# Compare numeric difference
if len(old_vec) == len(new_vec):
    diff = np.mean(np.abs(old_vec - new_vec))
    print("Mean absolute difference:", diff)
else:
    print("Different dimensions, new features are included")

samples = random.sample([f for f in os.listdir(new_dir) if f.endswith(".npy")], 5)
for file in samples:
    if not os.path.exists(os.path.join(old_dir, file)): continue
    old_vec = np.load(os.path.join(old_dir, file))
    new_vec = np.load(os.path.join(new_dir, file))
    if len(old_vec) == len(new_vec):
        diff = np.mean(np.abs(old_vec - new_vec))
        print(f"{file}: change in mean={diff:.4f}")
    else:
        print(f"{file}: shape changed ({len(old_vec)}→{len(new_vec)})")

import faiss

index = faiss.read_index("models/faiss_index.bin")
print("Number of vectors:", index.ntotal)
print("Vector dimension:", index.d)


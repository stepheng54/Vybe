import numpy as np
import librosa
import openl3

def get_openl3_embedding(path, duration=30.0):
    y, sr = librosa.load(path, sr=None, mono=True)
    if duration and len(y) > duration * sr:
        # center crop
        start = (len(y) - int(duration * sr)) // 2
        y = y[start:start + int(duration * sr)]
    emb, _ = openl3.get_audio_embedding(
        y, sr,
        content_type="music",
        embedding_size=512
    )
    return emb.mean(axis=0).astype("float32")

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

song_a = "C:\\Users\\sgilt\\OneDrive\\Desktop\\Vybe\\South Arcade - 2005.mp3"
song_b = "C:\\Users\\sgilt\\OneDrive\\Desktop\\Vybe\\data\\raw\\Personal\\South Arcade - FEAR OF HEIGHTS.mp3"

ea = get_openl3_embedding(song_a)
eb = get_openl3_embedding(song_b)

print("OpenL3 shapes:", ea.shape, eb.shape)
sim = cosine(ea, eb)
print("Cosine similarity (OpenL3):", sim)

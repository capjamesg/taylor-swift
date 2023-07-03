import librosa
import numpy as np

file1 = "../taylor-swift/data/vocals10.wav"
file2 = "../taylor-swift/data/vocals20.wav"

y1, sr1 = librosa.load(file1)
y2, sr2 = librosa.load(file2)

# get 

chroma1 = librosa.feature.chroma_stft(y1, sr1)
chroma2 = librosa.feature.chroma_stft(y2, sr2)

x_ref = librosa.feature.stack_memory(chroma1, n_steps=10, mode="edge")
x_test = librosa.feature.stack_memory(chroma2, n_steps=10, mode="edge")

# cross similarity
sim = librosa.segment.cross_similarity(x_ref, x_test, mode="distance", metric="cosine")

# get avg
distance = 0.0

for i in range(len(sim)):
    for j in range(len(sim[i])):
        distance += sim[i][j]

distance /= len(sim) * len(sim[0])

print(distance)
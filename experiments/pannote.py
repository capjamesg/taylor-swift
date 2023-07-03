import torch

import warnings
from scipy.spatial.distance import cdist

# supress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device="cuda" if torch.cuda.is_available() else "cpu",)

from pyannote.audio import Audio

audio = Audio(sample_rate=16000, mono="downmix")

waveform1, sample_rate = audio({"audio": "../taylor-swift/data/vocals30.wav"})
embedding1 = model(waveform1[None])

import os

all_sims = []

for file in os.listdir("../taylor-swift/data/"):
    if file.endswith(".wav") and "_m" not in file:
        waveform2, sample_rate = audio({"audio": "../taylor-swift/data/" + file})
        embedding2 = model(waveform2[None])

        # compare embeddings using "cosine" distance
        distance = cdist(embedding1, embedding2, metric="cosine")

        all_sims.append(distance[0][0])

avg = sum(all_sims) / len(all_sims)

print(avg)


import torch
import torchaudio
import matplotlib.pyplot as plt
from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade


def separate_sources(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def plot_spectrogram(stft, title="Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
    figure.suptitle(title)
    plt.colorbar(img, ax=axis)
    # save to file
    plt.savefig(f"{title}.png")

bundle = HDEMUCS_HIGH_MUSDB_PLUS

model = bundle.get_model()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

sample_rate = bundle.sample_rate

print(f"Sample rate: {sample_rate}")

# load from shakeitoff.opus
SAMPLE_SONG = "shakeitoff.opus"
waveform, sample_rate = torchaudio.load(SAMPLE_SONG)  # replace SAMPLE_SONG with desired path for different song
waveform = waveform.to(device)
mixture = waveform

# parameters
segment: int = 10
overlap = 0.1

print("Separating track")

ref = waveform.mean(0)
waveform = (waveform - ref.mean()) / ref.std()  # normalization

sources = separate_sources(
    model,
    waveform[None],
    device=device,
    segment=segment,
    overlap=overlap,
)[0]

print("Done separating track")

sources = sources * ref.std() + ref.mean()

sources_list = model.sources
sources = list(sources)

audios = dict(zip(sources_list, sources))

# get song length
length = mixture.shape[1] / sample_rate

# divide song into 10 second segments
length = int(length)

vocals_spec = audios["vocals"].cpu()

for i in range(0, length, 10):
    # get 10 second segment of vocals
    segment = vocals_spec[:, i * sample_rate : (i + 10) * sample_rate]

    # save segment as .wav file
    torchaudio.save(f"vocals_m{i}.wav", segment, sample_rate)

    # save spectrogram
    plot_spectrogram(segment, title=f"Vocals {i} Taylor Swift")


import os
import numpy as np
from PIL import Image
import random

# svm
from sklearn import svm

data = []
labels = []

id2name = {0: "Taylor", 1: "Not Taylor"}

# load all .jpg files in the directory
for file in os.listdir("."):
    if file.endswith(".png"):
        image = Image.open(file).convert("RGB")

        # convert image to numpy array
        if "2" in file:
            print(file)
            continue
        data.append(np.asarray(image))
        if "Taylor" in file:
            labels.append(1)
        else:
            labels.append(0)

model = svm.SVC(gamma=0.001, C=100)

# flatten data
n_samples = len(data)
data = np.array(data).reshape((n_samples, -1))

# train model
model.fit(data, labels)

test_data = Image.open("Vocals 220 Taylor Swift.png").convert("RGB")

# predict
print("Predicting...")

print("Taylor Swift" if model.predict(np.asarray(test_data).reshape(1, -1)) == 1 else "Not Taylor Swift")

# normalized cross correlation
import librosa
from scipy.spatial.distance import cosine
import warnings
import os
import numpy as np
# suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

CANONICAL_SONG = "vocals10.wav"
canonical_song_vector = []
SWIFT_SONGS = os.listdir("data/")

# sort songs alphabetically
SWIFT_SONGS.sort()

swift_vectors = []

for song in SWIFT_SONGS:
    y, sr = librosa.load(f"data/{song}")

    # if not 10 secs, skip
    if len(y) != 220500:
        continue

    mfcc = librosa.feature.mfcc(y, sr)
    swift_vectors.append(mfcc)
    if song == CANONICAL_SONG:
        print("Canonical song found")
        canonical_song_vector = mfcc

sims = {}

for song, i in zip(SWIFT_SONGS, range(len(SWIFT_SONGS))):
    distance = 0.0

    canonical = np.array(canonical_song_vector)

    print(i)

    if i > len(swift_vectors) - 1:
        break

    # get the distance between each frame
    sim = librosa.segment.cross_similarity(swift_vectors[i], canonical, mode="distance", metric="cosine")

    # get the average distance between each frame
    for i in range(len(sim)):
        for j in range(len(sim[i])):
            distance += sim[i][j]

    distance /= len(sim) * len(sim[0])

    sims[song] = distance

# order sims from smallest to largest
sims = {k: v for k, v in sorted(sims.items(), key=lambda item: item[1])}

# print sims

for song in sims:
    print(song, sims[song])
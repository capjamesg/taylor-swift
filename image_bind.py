import data
import torch
from models import imagebind_model
from models.imagebind_model import ModalityType
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import tempfile
import librosa
import numpy as np

app = Flask(__name__, static_folder="./static")

def get_embedding(audio):
    audio = {
        ModalityType.AUDIO: data.load_and_transform_audio_data([audio], device),
    }
    audio_embedding = model(audio)

    return audio_embedding[ModalityType.AUDIO]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

vectors = []
vector_names = []

with torch.no_grad():
    for file in os.listdir("./data/"):
        # if ends in wav
        if file.endswith(".wav") and "_" not in file:
            a = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(["./data/" + file], device),
            }
            a_embedding = model(a)

            vectors.append(a_embedding[ModalityType.AUDIO])
            vector_names.append(file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # get body from request audio/ogg; codecs=opus
        audio = request.files["file"]
        # get similarity
        with torch.no_grad():
                # uses torchvision to load audio, transform as necessary please
            # audio = n
            # create tmp file

            # create tmp file
            tmp = tempfile.NamedTemporaryFile()
            audio.save(tmp.name)

            # clean up audio with librosa
            # y, sr = librosa.load(tmp.name)

            # # remove silence
            # y, _ = librosa.effects.trim(y)

            # # convert to mono
            # y = librosa.to_mono(y)

            # # save as wav
            # librosa.output.write_wav(tmp.name, y, sr)
            
            audio = {
                ModalityType.AUDIO: data.load_and_transform_audio_data([tmp.name], device),
            }

            audio_embedding = model(audio)

            sims = []

            for i, v in enumerate(vectors):
                sim = torch.cosine_similarity(audio_embedding[ModalityType.AUDIO], v)
                print(sim, vector_names[i])
                sims.append(sim)

            # print file name of most similar vector
            print(os.listdir("./data/")[sims.index(max(sims))])

            # print avg. sim
            avg_sim = sum(sims) / len(sims)

            avg_sim = avg_sim.cpu().numpy()

            # cast from float32 to float py
            avg_sim = avg_sim.astype(float)
            
            # round to 2 decimal places
            avg_sim = np.round(avg_sim, 2)

            print(avg_sim)

        # return similarity
        return jsonify({"similarity": avg_sim[0]})
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8084)
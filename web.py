import data
import torch
# from models import imagebind_model
# from models.imagebind_model import ModalityType
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

def get_embedding(audio):
    audio = {
        ModalityType.AUDIO: data.load_and_transform_audio_data([audio], device),
    }
    audio_embedding = model(audio)

    return audio_embedding[ModalityType.AUDIO]

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Instantiate model
# model = imagebind_model.imagebind_huge(pretrained=True)
# model.eval()
# model.to(device)

vectors = []

with torch.no_grad():
    for file in os.listdir("../"):
        # if ends in wav
        if file.endswith(".wav") and "_" not in file:
            a = {
                ModalityType.AUDIO: data.load_and_transform_audio_data(["../" + file], device),
            }
            a_embedding = model(a)

            vectors.append(a_embedding[ModalityType.AUDIO])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print(request.data)
        # get body from request audio/wav, as blb
        print(request.files)
        file = request.files['file']

        

        import librosa

        # plot spectrogram
        import matplotlib.pyplot as plt
        import librosa.display
        import numpy as np

        y, sr = librosa.load(file)

        # if length is > 10 seconds, return error
        # plot as spectrogram
        
        # if audio.length > 10:
        #     return render_template("index.html", error="Audio must be less than 10 seconds")

        # get similarity
        with torch.no_grad():
            audio = {
                ModalityType.AUDIO: data.load_and_transform_audio_data([audio], device),
            }

            audio_embedding = model(audio)

            sims = []

            for v in vectors:
                sim = torch.cosine_similarity(audio_embedding[ModalityType.AUDIO], v)
                sims.append(sim)

            # print avg. sim
            print(sum(sims) / len(sims))

        # return similarity
        return render_template("index.html", sim=sim)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
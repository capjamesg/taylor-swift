import data
import torch
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import tempfile
import librosa
import numpy as np

import warnings

# supress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

from speechbrain.pretrained import SpeakerRecognition

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)


app = Flask(__name__, static_folder="./static")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # get body from request audio/ogg; codecs=opus
        audio = request.files["file"]
        # get similarity
        with torch.no_grad():
            # create tmp file
            tmp = tempfile.NamedTemporaryFile()
            audio.save(tmp.name)

            import subprocess

            import random
            file_name = str(random.randint(0, 10000)) + ".wav"

            subprocess.call(
                [
                    "ffmpeg",
                    "-i",
                    tmp.name,
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "./" + file_name,
                ]
            )

            all_sims = []

            for file in os.listdir("./data/")[:5]:
                if file.endswith(".wav") and "_" not in file:
                    score, prediction = verification.verify_files(
                        file_name, "./data/" + file
                    )

                    print(score, prediction)

                    if prediction == False:
                        continue

                    if score == 1:
                        continue

                    print(file, score)

                    all_sims.append(score)

            os.remove("./" + file_name)

            if len(all_sims) == 0:
                avg = 0
            else:
                avg = max(all_sims)

            # round to 1 decimal place
            avg_sim = np.round(avg, 1)

        # return similarity
        return jsonify({"similarity": avg_sim.tolist()})
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8084)
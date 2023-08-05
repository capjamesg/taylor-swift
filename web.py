import torch
import os
from flask import Flask, render_template, request, jsonify
import tempfile
import librosa
import json
import numpy as np
import random
import subprocess

import warnings

# supress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

from speechbrain.pretrained import SpeakerRecognition

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)

if not os.path.exists("leaderboard.json"):
    with open("leaderboard.json", "w") as f:
        json.dump([], f)

with open("leaderboard.json", "r") as f:
    leaderboard = json.load(f)
    

app = Flask(__name__, static_folder="./static")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # get body from request audio/ogg; codecs=opus
        audio = request.files["file"]
        username = request.form["username"]

        usernames = set([entry["username"] for entry in leaderboard])

        while username in usernames:
            username += str(random.randint(0, 9))

        # get similarity
        with torch.no_grad():
            # create tmp file
            tmp = tempfile.NamedTemporaryFile()
            audio.save(tmp.name)

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

                    if prediction == False:
                        continue

                    if score == 1:
                        continue

                    all_sims.append(score)

            os.remove("./" + file_name)

            if len(all_sims) == 0:
                avg = 0
            else:
                avg = max(all_sims)

            # round to 1 decimal place
            avg_sim = np.round(avg, 1)

            # cast to int
            avg_sim = int(avg_sim)

            if username:
                # leaderboard is ordered by score, so we need to insert at the position where the score is less than the current score
                found = False
                for i, entry in enumerate(leaderboard):
                    if entry["similarity"] <= avg_sim:
                        leaderboard.insert(i, {"username": username, "similarity": avg_sim})
                        found = True
                        break

                if not found:
                    leaderboard.append({"username": username, "similarity": avg_sim})

                with open("leaderboard.json", "w") as f:
                    json.dump(leaderboard, f)

        return jsonify({"similarity": avg_sim, "leaderboard": leaderboard})
    
    # add rank to leaderboard
    leaderboard_with_rank = [
        {"username": entry["username"], "similarity": entry["similarity"], "rank": i + 1}
        for i, entry in enumerate(leaderboard)
    ]
    
    return render_template("index.html", leaderboard=leaderboard_with_rank)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8084)
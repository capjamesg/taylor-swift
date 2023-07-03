# swifties.me

[swifties.me](https://swifties.me) is a web application that lets you find out how similar your voice is to Taylor Swift. Swifties offers two voice comparison methods: (i) using ImageBind audio embeddings, and; (ii) using the SpeechBrain speaker verification model.

More investigation is needed to determine which method is more accurate. It is not clear whether ImageBind is good for speaker identification: in testing, qualitatively, ImageBind ranked Taylor Swift speaking higher than non-Taylor Swift samples. But, there were false positives. This was the case for SpeechBrain. It may be the case that another approach would achieve better results.

## How it Works

Both the ImageBind and SpeechBrain approach accept an arbitrary number of clips of Taylor Swift singing. These clips need to have isolated vocals so that they represent Swift singing rather than the audio in the background. [Demucs] by Meta Research proved effective in this task.

When a user records an audio clip, it is compared against either:

1. Pre-computed ImageBind embeddings for the isolated vocal clips, or;
2. The raw vocal clips using the SpeechBrain speaker verification model.

A similarity score is then returned to the user.

## Getting Started

### ImageBind

First, you will need to install ImageBind. You can do this by following the official ImageBind installation instructions.

In the ImageBind root folder, copy the web app in this repository (`image_bind.py`). Then, install all the requirements for this project:

pip install -r ts-requirements.txt

Then, run the web app in debugging mode:

python3 web.py

### SpeechBrain

First, install the dependencies for the project:

pip install -r ts-requirements.txt

Then, run the web app in debugging mode:

python3 web.py

## Using the App

After running the `web.py` script, the web application will be available at `localhost:8084`. You will need to run the app on a HTTPS server as browser audio recording requires a secure context.

## Experiments

The `experiments` folder contains miscellaneous code not used in the application.

## License

This project is licensed under an [MIT license](LICENSE).

## Acknowledgements

- The HTML and CSS for this app were started in [Glitch](https://glitch.com/). Thank you Glitch for making an intuitive web interface for HTML editing.
- ImageBind and SpeechBrain for their open source projects.
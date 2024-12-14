from datetime import datetime

import whisper
from flask import Flask

app = Flask(__name__)

model = whisper.load_model("turbo")


@app.route("/")
def hello_world():
    result = model.transcribe("whisper-example-audio.mp3")
    return result["text"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

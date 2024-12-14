from datetime import datetime

import whisper
from flask import Flask

app = Flask(__name__)

print("loading model...")
model = whisper.load_model("tiny", download_root="/whisperdata")
print("... model loaded!!")


@app.route("/")
def hello_world():
    print("index page!! transcribing")
    result = model.transcribe("whisper-example-audio.mp3")
    print(result)
    print(result["text"])
    print("transcribed!!")
    return result["text"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

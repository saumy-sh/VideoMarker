from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
from pydub import AudioSegment
import numpy as np
from vosk import Model, KaldiRecognizer
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=embedder, verbose=True)

# STT Model
model_path = "./stt/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"
recogniser = KaldiRecognizer(Model(model_path), 16000)

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed
VIDEO_DIR = "static/videos"
AUDIO_DIR = "static/audio"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # --- Save uploaded video ---
        file = request.files["video"]
        video_path = os.path.join(VIDEO_DIR, file.filename)
        file.save(video_path)

        # --- Extract audio ---
        try:
            audio = AudioSegment.from_file(video_path, format="mp4")
            audio = audio.set_channels(1).set_frame_rate(16000)

            # Normalize
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / np.iinfo(audio.array_type).max
            duration_sec = len(audio) / 1000
        except Exception as e:
            return f"Error extracting audio: {e}"

        # --- Split into 20s chunks ---
        try:
            chunk_length_ms = 20 * 1000
            chunks_metadata = []

            for i in range(0, len(audio), chunk_length_ms):
                start_ms = i
                end_ms = min(i + chunk_length_ms, len(audio))
                chunk = audio[start_ms:end_ms]
                # pass it to whisper for STT and then store topics for each audio only in session because sessions can't
                # store audio segments
                samples = np.array(chunk.get_array_of_samples(), dtype=np.int16)
                recogniser.AcceptWaveform(samples.tobytes())
                text = json.loads(recogniser.Result()['text'])
                
                start_sec = start_ms / 1000
                end_sec = end_ms / 1000

                chunks_metadata.append({
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "text": text
                })
            docs = [chunk["text"] for chunk in chunks_metadata]
            print(docs)
            topics, _ = topic_model.fit_transform(docs)
            print(topic_model.get_topic_info())
            session['chunks_metadata'] = chunks_metadata
            

        except Exception as e:
            return f"Error splitting audio: {e}"

        return render_template(
                "index.html",
                video_uploaded=True,
                video_file=file.filename
            )

    return render_template("index.html", video_uploaded=False)

# query video route
@app.route("/query", methods=["POST"])
def query_video():
    if request.method == "POST":
        chunks_metadata = session.get('chunks_metadata')
        print(chunks_metadata)
    query_text = request.form["query"]
    return jsonify({"result": f"You searched for: {query_text}"})


if __name__ == "__main__":
    app.run(port=5500,debug=True)

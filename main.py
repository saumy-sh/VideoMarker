from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
from pydub import AudioSegment
import numpy as np
from vosk import Model, KaldiRecognizer
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# STT Model
MODEL_PATH = "./stt/vosk-model-en-us-0.22/vosk-model-en-us-0.22"
recogniser = KaldiRecognizer(Model(MODEL_PATH), 16000)

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed
VIDEO_DIR = "static/videos"
AUDIO_DIR = "static/audio"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session.clear()
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
                result = json.loads(recogniser.Result())
                text = result.get("text", "")
                start_sec = start_ms / 1000
                end_sec = end_ms / 1000

                chunks_metadata.append({
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "text": text
                })
            # collect non-empty texts
            docs = [chunk["text"] for chunk in chunks_metadata if chunk["text"].strip()]

            # Topic modelling (LDA)
            vectorizer = CountVectorizer(stop_words="english")
            X = vectorizer.fit_transform(docs)

            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(X)

            # Extract top words for each topic
            n_top_words = 5
            topic_keywords = {}
            for idx, topic in enumerate(lda.components_):
                top_features = topic.argsort()[-n_top_words:]
                topic_keywords[idx] = [vectorizer.get_feature_names_out()[i] for i in top_features]

            # Map each doc to its best topic
            doc_topics = lda.transform(X)
            doc_best_topics = doc_topics.argmax(axis=1)

            # Replace text with topic keywords
            doc_idx = 0
            for chunk in chunks_metadata:
                if chunk["text"].strip():
                    topic_id = doc_best_topics[doc_idx]
                    chunk["topics"] = topic_keywords[topic_id]   # store list of words
                    doc_idx += 1
                del chunk["text"]  # remove raw text

            # Save in session
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
        if not chunks_metadata:
            return jsonify({"error": "No video uploaded or processed."}), 400
        

        
    query_text = request.form["query"]
    return jsonify({"result": f"You searched for: {query_text}"})


if __name__ == "__main__":
    app.run(port=5500,debug=True, use_reloader=False)

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
from pydub import AudioSegment
import numpy as np
from vosk import Model, KaldiRecognizer
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# keyword extraction model
kw_model = KeyBERT()

# chunking size
CHUNK_SIZE = 30  # seconds

# wav2Vec2 Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

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

        # --- Split into 120s chunks ---
        try:
            chunk_length_ms = CHUNK_SIZE * 1000
            chunks_metadata = []

            for i in range(0, len(audio), chunk_length_ms):
                start_ms = i
                end_ms = min(i + chunk_length_ms, len(audio))
                chunk = audio[start_ms:end_ms]
                inputs = processor(chunk, sampling_rate=audio.sample_rate, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model(**inputs).logits
                text = processor.decode(torch.argmax(logits, dim=-1)[0])
                print(text)
                # pass it to whisper for STT and then store topics for each audio only in session because sessions can't
                # store audio segments
                # samples = np.array(chunk.get_array_of_samples(), dtype=np.int16)
                # recogniser.AcceptWaveform(samples.tobytes())
                # result = json.loads(recogniser.Result())
                # text = result.get("text", "")
                keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english')
                print(keywords)
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
        session["video_file"] = file.filename
        session["video_duration"] = duration_sec
        return render_template(
                "index.html",
                video_uploaded=True,
                video_file=file.filename,
                duration=duration_sec
            )

    return render_template("index.html", video_uploaded=False)

# query video route
@app.route("/query", methods=["POST"])
def query_video():
    if request.method == "POST":
        chunks_metadata = session.get('chunks_metadata')
        if not chunks_metadata:
            return jsonify({"error": "No video uploaded or processed."}), 400

        query = request.form.get("query", "").strip().lower()
        if not query:
            return jsonify({"error": "No query provided."}), 400

        # Prepare docs from chunks
        docs = [" ".join(chunk["topics"]) for chunk in chunks_metadata]

        # Vectorizer (reuse from training LDA)
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(docs)

        # Fit/reuse LDA on chunks
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)

        # Transform query into topic distribution
        query_vec = vectorizer.transform([query])
        query_topic_dist = lda.transform(query_vec)[0]

        # Pick the dominant topic for query
        query_topic_idx = query_topic_dist.argmax()
        query_topic_words = [
            vectorizer.get_feature_names_out()[i]
            for i in lda.components_[query_topic_idx].argsort()[-10:]
        ]

        # Convert all topics (chunks + query) into docs
        query_doc = " ".join(query_topic_words)
        docs_with_query = docs + [query_doc]

        # TF-IDF similarity
        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs_with_query)

        query_vec = tfidf_matrix[-1]  # last row = query
        doc_vecs = tfidf_matrix[:-1]  # all chunks
        similarities = cosine_similarity(query_vec, doc_vecs)[0]

        # Find best match
        best_idx = similarities.argmax()
        best_chunk = chunks_metadata[best_idx]

        # Send result back to template
        return render_template(
            "index.html",
            video_uploaded=True,
            video_file=session.get("video_file"),
            duration=session.get("video_duration"),
            query=query,
            matched_topics=best_chunk["topics"],
            start_sec=best_chunk["start_sec"],
            end_sec=best_chunk["end_sec"]
        )
        
    query_text = request.form["query"]
    return jsonify({"result": f"You searched for: {query_text}"})


if __name__ == "__main__":
    app.run(port=5500,debug=True, use_reloader=False)

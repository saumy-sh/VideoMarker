from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import os
from pydub import AudioSegment
import numpy as np
# from vosk import Model, KaldiRecognizer
import json
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from keyword_generator import generate_keywords,similar_keywords
import uuid
from summariser_embedder import generate_embeddings,most_similar_text


# chunking size
CHUNK_SIZE = 30  # seconds

# metadata folder

# wav2Vec2 Model
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# # processor = processor.to(device)
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# model = model.to(device)

# STT Model
# MODEL_PATH = "./stt/vosk-model-en-us-0.22/vosk-model-en-us-0.22"
# recogniser = KaldiRecognizer(Model(MODEL_PATH), 16000)

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed
VIDEO_DIR = "static/videos"
AUDIO_DIR = "static/audio"
METADATA_DIR = "metadata"
os.makedirs(METADATA_DIR, exist_ok=True)
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


        
        # try:
        # keywords_dict = generate_keywords(audio,CHUNK_SIZE) 
        file_id = str(uuid.uuid4())
        file_path = os.path.join(METADATA_DIR, f"{file_id}.faiss")
        generate_embeddings(audio,file_path,CHUNK_SIZE)
        # with open(file_path,"w") as f:
        #     json.dump(keywords_dict,f)
        session["metadata"] = file_id
        # except Exception as e:
        #     return f"Error splitting audio: {e}"
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
        file_id = session['metadata']
        file_path = os.path.join(METADATA_DIR, f"{file_id}.faiss")
        print(file_path)
        # with open(file_path,"r") as f:
        #     keywords_dict = json.load(f)
        
        print("==============================")
        # print(keywords_dict)
        # if not keywords_dict:
        #     return jsonify({"error": "No video uploaded or processed."}), 400

        query = request.form.get("query", "").strip().lower()
        if not query:
            return jsonify({"error": "No query provided."}), 400
        time_starts = most_similar_text(query,file_path,CHUNK_SIZE)
        # time_starts = similar_keywords(query,keywords_dict)
        
        # # TF-IDF similarity
        # tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        # tfidf_matrix = tfidf_vectorizer.fit_transform(docs_with_query)

        # query_vec = tfidf_matrix[-1]  # last row = query
        # doc_vecs = tfidf_matrix[:-1]  # all chunks
        # similarities = cosine_similarity(query_vec, doc_vecs)[0]

        # # Find best match
        # best_idx = similarities.argmax()
        # best_chunk = chunks_metadata[best_idx]

        # Send result back to template
        return render_template(
            "index.html",
            video_uploaded=True,
            video_file=session.get("video_file"),
            duration=session.get("video_duration"),
            query=query,
            # matched_topics=best_chunk["topics"],
            start_secs=[time_starts],
        )
        
    query_text = request.form["query"]
    return jsonify({"result": f"You searched for: {query_text}"})


if __name__ == "__main__":
    app.run(port=5500,debug=True, use_reloader=False)

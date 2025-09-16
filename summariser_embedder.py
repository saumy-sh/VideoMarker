from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import faiss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)

# Embedding model
summariser_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

def generate_embeddings(audio,embedding_path, CHUNK_SIZE=30):
    sample_rate = audio.frame_rate
    chunk_length_ms = CHUNK_SIZE * 1000
    asr_texts = []

    for i in range(0, len(audio), chunk_length_ms):
        start_ms = i
        end_ms = min(i + chunk_length_ms, len(audio))
        chunk = audio[start_ms:end_ms]

        # Convert to float32, normalize [-1,1]
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0

        inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        asr_texts.append(text)

        # start_sec = start_ms / 1000
        

    # Encode transcriptions into embeddings
    embeddings = summariser_model.encode(asr_texts, convert_to_tensor=True, device=device)
    emb_np = embeddings.cpu().numpy()
    # Build FAISS index
    index = faiss.IndexFlatL2(emb_np.shape[1])
    index.add(emb_np)

    faiss.write_index(index, embedding_path)
    print("===EMBEDDINGS CREATED===")
    return

def most_similar_text(query,file_path,chunk_size):
    query_embedding = summariser_model.encode([query])
    index = faiss.read_index(file_path)
    D, I = index.search(query_embedding, k=1)
    best_idx = I[0][0]
    return int(best_idx*chunk_size)
    


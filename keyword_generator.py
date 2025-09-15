from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from keybert import KeyBERT
import numpy as np
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model = model.to(device)

# keyword extraction model
kw_model = KeyBERT()


def generate_keywords(audio,CHUNK_SIZE):
    keywords_dict = defaultdict(set)
    sample_rate = audio.frame_rate
    chunk_length_ms = CHUNK_SIZE * 1000

    for i in range(0, len(audio), chunk_length_ms):
        start_ms = i
        end_ms = min(i + chunk_length_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
        inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(text)
        
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english')
        start_sec = start_ms / 1000
        for keyword in keywords:
            if keyword:
                keywords_dict[keyword[0]].add(float(start_sec))
        session_keywords = {k: list(v) for k, v in keywords_dict.items()}
        print(session_keywords)

    return session_keywords
        
def similar_keywords(text,keywords_dict):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english')
    time_starts = set()
    for key in keywords:
        time_starts = time_starts.union(keywords_dict[key[0]])
    return list(time_starts)
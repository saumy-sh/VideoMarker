# chunk_length_ms = CHUNK_SIZE * 1000
#             chunks_metadata = []
            
#             for i in range(0, len(audio), chunk_length_ms):
#                 start_ms = i
#                 end_ms = min(i + chunk_length_ms, len(audio))
#                 chunk = audio[start_ms:end_ms]
#                 samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0  # normalize
#                 inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt").to(device)

#                 with torch.no_grad():
#                     generated_ids = model.generate(**inputs)

#                 text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#                 print(text)
#                 # samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
#                 # inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt", padding=True)
#                 # # Move inputs to GPU/CPU depending on model
#                 # inputs = {k: v.to(device) for k, v in inputs.items()}
#                 # with torch.no_grad():
#                 #     logits = model(**inputs).logits
#                 # text = processor.decode(torch.argmax(logits, dim=-1)[0])
#                 # print(text)
#                 # pass it to whisper for STT and then store topics for each audio only in session because sessions can't
#                 # store audio segments
#                 # samples = np.array(chunk.get_array_of_samples(), dtype=np.int16)
#                 # recogniser.AcceptWaveform(samples.tobytes())
#                 # result = json.loads(recogniser.Result())
#                 # text = result.get("text", "")
#                 keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english')
#                 print(keywords)
#                 start_sec = start_ms / 1000
#                 end_sec = end_ms / 1000

#                 chunks_metadata.append({
#                     "start_sec": float(start_sec),
#                     "end_sec": float(end_sec),
#                     "text": text
#                 })
#             # collect non-empty texts
#             docs = [chunk["text"] for chunk in chunks_metadata if chunk["text"].strip()]

#             # Topic modelling (LDA)
#             vectorizer = CountVectorizer(stop_words="english")
#             X = vectorizer.fit_transform(docs)

#             lda = LatentDirichletAllocation(n_components=5, random_state=42)
#             lda.fit(X)

#             # Extract top words for each topic
#             n_top_words = 5
#             topic_keywords = {}
#             for idx, topic in enumerate(lda.components_):
#                 top_features = topic.argsort()[-n_top_words:]
#                 topic_keywords[idx] = [vectorizer.get_feature_names_out()[i] for i in top_features]

#             # Map each doc to its best topic
#             doc_topics = lda.transform(X)
#             doc_best_topics = doc_topics.argmax(axis=1)

#             # Replace text with topic keywords
#             doc_idx = 0
#             for chunk in chunks_metadata:
#                 if chunk["text"].strip():
#                     topic_id = doc_best_topics[doc_idx]
#                     chunk["topics"] = topic_keywords[topic_id]   # store list of words
#                     doc_idx += 1
#                 del chunk["text"]  # remove raw text

#             # Save in session
#             session['chunks_metadata'] = chunks_metadata
        
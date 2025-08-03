import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle

class CaptionIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []  # List of dicts
        self.dimension = self.embedder.get_sentence_embedding_dimension()

    def build_index(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        captions = [item["caption"] for item in data]
        embeddings = self.embedder.encode(captions, convert_to_numpy=True)

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))
        self.metadata = data  

        print(f"Index built with {len(captions)} captions.")

    def save_index(self, faiss_path="captions.index", meta_path="metadata.pkl"):
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved FAISS index to '{faiss_path}' and metadata to '{meta_path}'.")

    def load_index(self, faiss_path="captions.index", meta_path="metadata.pkl"):
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Loaded FAISS index and metadata.")

    def search(self, query, top_k=5):
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(np.array(query_vec), top_k)

        results = []
        for idx in indices[0]:
            results.append(self.metadata[idx])
        return results


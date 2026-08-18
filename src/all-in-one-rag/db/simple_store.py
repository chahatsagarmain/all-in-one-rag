import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from models.vector_store import VectorStore

class SimpleVectorStore(VectorStore):
    def __init__(self, storage_path: str = "./data/vector_store.pkl"):
        self.storage_path = storage_path
        self.collections: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as f:
                    self.collections = pickle.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load local vector store: {e}")
                self.collections = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, "wb") as f:
                pickle.dump(self.collections, f)
        except Exception as e:
            print(f"[Error] Failed to save local vector store: {e}")

    def create_collection(self, collection_name: str, vector_dim: int) -> None:
        self.collections[collection_name] = {
            "documents": [],
            "embeddings": [],
            "metadata": [],
            "vector_dim": vector_dim
        }
        self._save()

    def add_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict[str, Any]]
    ) -> None:
        if collection_name not in self.collections:
            vector_dim = len(embeddings[0]) if embeddings else 0
            self.create_collection(collection_name, vector_dim)

        self.collections[collection_name]["documents"].extend(documents)
        self.collections[collection_name]["embeddings"].extend(embeddings)
        self.collections[collection_name]["metadata"].extend(metadata)
        self._save()

    def similarity_search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if collection_name not in self.collections:
            return []

        col = self.collections[collection_name]
        if not col["embeddings"]:
            return []

        # Cosine similarity using NumPy
        emb_matrix = np.array(col["embeddings"])
        qv = np.array(query_vector)
        
        # Calculate dot products and norms
        dot_products = np.dot(emb_matrix, qv)
        matrix_norms = np.linalg.norm(emb_matrix, axis=1)
        qv_norm = np.linalg.norm(qv)
        
        # Avoid division by zero
        norms = matrix_norms * qv_norm
        norms[norms == 0] = 1e-10
        
        similarities = dot_products / norms
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": col["documents"][idx],
                "metadata": col["metadata"][idx],
                "distance": float(1.0 - similarities[idx])  # Cosine distance
            })
        return results

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_vector: List[float],
        alpha: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        col = self.collections.get(collection_name)
        if not col or not col["documents"]:
            return []

        # Retrieve all similarities first
        all_docs_count = len(col["documents"])
        sim_results = self.similarity_search(collection_name, query_vector, top_k=all_docs_count)
        if not sim_results:
            return []

        documents = col["documents"]

        # Simple term overlap (keyword score)
        query_words = set(query_text.lower().split())
        keyword_scores = []
        for doc in documents:
            doc_words = doc.lower().split()
            if not doc_words:
                keyword_scores.append(0.0)
                continue
            overlap = sum(1 for w in query_words if w in doc_words)
            keyword_scores.append(overlap / len(query_words) if query_words else 0.0)

        # Merge scores
        merged_results = []
        for item in sim_results:
            # Locate original document index
            try:
                original_idx = documents.index(item["text"])
            except ValueError:
                continue
            semantic_score = 1.0 - item["distance"]  # Convert distance back to similarity
            keyword_score = keyword_scores[original_idx]
            
            combined_score = alpha * semantic_score + (1.0 - alpha) * keyword_score
            merged_results.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "score": combined_score
            })

        # Sort by combined score descending
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        return merged_results[:top_k]

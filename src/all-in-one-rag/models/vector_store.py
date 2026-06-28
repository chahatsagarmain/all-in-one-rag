from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):

    @abstractmethod
    def create_collection(self, collection_name: str, vector_dim: int) -> None:
        """Create a collection/class in the vector store with a specified vector dimension."""
        pass

    @abstractmethod
    def add_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Add documents/chunks along with their embeddings and metadata to the collection."""
        pass

    @abstractmethod
    def similarity_search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform a similarity search using a query vector and return the top_k matching documents."""
        pass

    @abstractmethod
    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_vector: List[float],
        alpha: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid search using both query text (keyword) and query vector (semantic)."""
        pass

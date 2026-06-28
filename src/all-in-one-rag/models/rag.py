from abc import ABC, abstractmethod

class AbstractRAGPipeline(ABC):

    @abstractmethod
    def build(self):
        """Build and execute the RAG pipeline."""
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5) -> list:
        """Query the RAG pipeline for relevant documents/context chunks."""
        pass

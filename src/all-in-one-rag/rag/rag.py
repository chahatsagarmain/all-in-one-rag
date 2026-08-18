from models.ingestor import Ingester
from models.rag import AbstractRAGPipeline
from typing import Type, Dict, Any
from models.chunker import Chunker
from models.embedder import Embedder
from models.vector_store import VectorStore

class RAG(AbstractRAGPipeline):

    def __init__(
        self, 
        path: str, 
        ingestor: Ingester, 
        chunker_class: Type[Chunker] | None = None, 
        embedder_class: Type[Embedder] | None = None,
        vector_store_class: Type[VectorStore] | None = None
    ):
        self.__ingestor = ingestor
        self.__chunker_class = chunker_class
        self.__embedder_class = embedder_class
        self._path = path
        
        if vector_store_class is None:
            from db.weaviate_store import WeaviateVectorStore
            self.__store = WeaviateVectorStore()
        else:
            self.__store = vector_store_class()
    
    def build(self):
        content = self.__ingestor.ingest_data_from_path(self._path)
        print("Ingested content successfully.")
        
        chunks_data: Dict[str, Any] | None = None
        if self.__chunker_class is not None:
            chunker = self.__chunker_class(content)
            chunks_data = chunker.create_chunks()
            print(f"Successfully chunked content into {len(chunks_data['chunks'])} chunks.")
        
        if chunks_data is not None and self.__embedder_class is not None:
            embedder = self.__embedder_class(chunks_data)
            embeddings_data = embedder.create_embeddings()
            if embeddings_data is None:
                return
            embeddings = embeddings_data["embeddings"]
            print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}.")
            
            try:
                collection_name = "DocumentChunk"
                vector_dim = len(embeddings[0]) if embeddings else 0
                
                if vector_dim > 0:
                    self.__store.create_collection(collection_name, vector_dim)
                    metadata_list = [chunks_data["metadata"]] * len(chunks_data["chunks"])
                    self.__store.add_documents(
                        collection_name=collection_name,
                        documents=chunks_data["chunks"],
                        embeddings=embeddings,
                        metadata=metadata_list
                    )
                    print(f"Successfully persisted all chunks in {self.__store.__class__.__name__}!")
                else:
                    print("Skipping storage: No embeddings generated.")
            except Exception as e:
                print(f"\n[Warning] Could not persist data in vector store: {e}")
                print("If using Weaviate, make sure your docker container is running (use 'docker compose up -d').\n")

    def query(self, query_text: str, top_k: int = 5) -> list:
        if self.__embedder_class is None:
            print("No embedder class configured. Cannot run retrieval.")
            return []
            
        embedder = self.__embedder_class({"chunks": [query_text], "metadata": {}})
        embeddings_data = embedder.create_embeddings()
        embeddings = embeddings_data.get("embeddings", [])
        if not embeddings:
            return []
        query_vector = embeddings[0]
        
        try:
            collection_name = "DocumentChunk"
            results = self.__store.hybrid_search(collection_name, query_text , query_vector, top_k=top_k)
            return results
        except Exception as e:
            print(f"Error retrieving from Weaviate: {e}")
            return []

import json
from typing import List, Dict, Any
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from models.vector_store import VectorStore

class WeaviateVectorStore(VectorStore):

    def __init__(self, host: str = "localhost", port: int = 8080, grpc_port: int = 50051):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self._client = None

    def connect(self) -> weaviate.WeaviateClient:
        """Establish connection to Weaviate local instance if not already connected."""
        if self._client is None:
            try:
                self._client = weaviate.connect_to_local(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port
                )
            except Exception as e:
                print(f"Error connecting to Weaviate: {e}")
                raise e
        return self._client

    def close(self) -> None:
        """Close connection to Weaviate."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def create_collection(self, collection_name: str, vector_dim: int) -> None:
        client = self.connect()
        
        # Weaviate collections must be capitalized and match the pattern [A-Z][a-zA-Z0-9_]*
        collection_name = collection_name[0].upper() + collection_name[1:]
        
        # Check if collection already exists
        if client.collections.exists(collection_name):
            print(f"Collection '{collection_name}' already exists. Recreating it to align vector dimensions.")
            client.collections.delete(collection_name)

        client.collections.create(
            name=collection_name,
            vectorizer_config=None,  # We generate vectors ourselves
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
            ],
            vector_index_config=Configure.VectorIndex.hnsw()
        )
        print(f"Collection '{collection_name}' created successfully with vector dimension {vector_dim}.")

    def add_documents(
        self, 
        collection_name: str, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadata: List[Dict[str, Any]]
    ) -> None:
        client = self.connect()
        collection_name = collection_name[0].upper() + collection_name[1:]
        
        collection = client.collections.get(collection_name)
        
        print(f"Batch uploading {len(documents)} objects to Weaviate collection '{collection_name}'...")
        with collection.batch.dynamic() as batch:
            for doc, vector, meta in zip(documents, embeddings, metadata):
                batch.add_object(
                    properties={
                        "text": doc,
                        "metadata": json.dumps(meta)
                    },
                    vector=vector
                )
        print("Batch upload finished successfully.")

    def similarity_search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        client = self.connect()
        collection_name = collection_name[0].upper() + collection_name[1:]
        
        collection = client.collections.get(collection_name)
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=["distance"]
        )
        
        results = []
        for obj in response.objects:
            results.append({
                "text": obj.properties.get("text"),
                "metadata": json.loads(obj.properties.get("metadata", "{}")),
                "distance": obj.metadata.distance if obj.metadata else None
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
        client = self.connect()
        collection_name = collection_name[0].upper() + collection_name[1:]
        
        collection = client.collections.get(collection_name)
        
        response = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,
            limit=top_k,
            return_metadata=["score"]
        )
        
        results = []
        for obj in response.objects:
            results.append({
                "text": obj.properties.get("text"),
                "metadata": json.loads(obj.properties.get("metadata", "{}")),
                "score": obj.metadata.score if obj.metadata else None
            })
        return results

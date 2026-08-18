from typing import Type, Union
from ingester.ingester import DataSourceIngester , AudioIngester
from chunker.chunker import FixedSizeChunker, SemanticChunker
from embedder.embedder import LocalEmbedder, OpenAIEmbedder, StaticEmbedder
from config.config import Config
from rag.rag import RAG
from models.rag import AbstractRAGPipeline
from models.chunker import Chunker
from models.embedder import Embedder


from models.vector_store import VectorStore


class RAGBuilder:

    def __init__(self):
        self._ingestor = None
        self._chunker_class = None
        self._embedder_class = None
        self._vector_store_class = None
        self._path = ""

    def with_data_source(self, file_path: str) -> "RAGBuilder":
        ingestor_type = file_path.split(".")[-1]
        if ingestor_type in ["pdf", "txt", "md"]:
            self._ingestor = DataSourceIngester()
        elif ingestor_type in ["mp3" , "wav" , "webm"]:
            self._ingestor = AudioIngester()
        else:
            raise TypeError(f"Unsupported file format: {ingestor_type}")
        self._path = file_path
        return self

    def with_chunker(self, chunker: Union[str, Type[Chunker]]) -> "RAGBuilder":
        if isinstance(chunker, str):
            if chunker == "semantic":
                self._chunker_class = SemanticChunker
            elif chunker == "fixed":
                self._chunker_class = FixedSizeChunker
            else:
                raise TypeError(f"Unknown chunking method: {chunker}")
        else:
            self._chunker_class = chunker
        return self

    def with_embedder(self, embedder: Union[str, Type[Embedder]]) -> "RAGBuilder":
        if isinstance(embedder, str):
            if embedder == "local":
                self._embedder_class = LocalEmbedder
            elif embedder == "openai":
                self._embedder_class = OpenAIEmbedder
            elif embedder == "static":
                self._embedder_class = StaticEmbedder
            else:
                raise TypeError(f"Unknown embedding method: {embedder}")
        else:
            self._embedder_class = embedder
        return self

    def with_vector_store(self, vector_store: Union[str, Type[VectorStore]]) -> "RAGBuilder":
        if isinstance(vector_store, str):
            if vector_store == "weaviate":
                from db.weaviate_store import WeaviateVectorStore
                self._vector_store_class = WeaviateVectorStore
            elif vector_store in ["simple", "local_file"]:
                from db.simple_store import SimpleVectorStore
                self._vector_store_class = SimpleVectorStore
            else:
                raise TypeError(f"Unknown vector store method: {vector_store}")
        else:
            self._vector_store_class = vector_store
        return self

    def build(self) -> AbstractRAGPipeline:
        if self._ingestor is None:
            raise ValueError("Data source must be set before building the pipeline")

        config = Config()

        # Fallback to Config defaults if not explicitly configured
        chunker = self._chunker_class
        if chunker is None:
            if config.chunking_method == "semantic":
                chunker = SemanticChunker
            elif config.chunking_method == "fixed":
                chunker = FixedSizeChunker
            else:
                raise TypeError(f"Unknown chunking method in Config: {config.chunking_method}")

        embedder = self._embedder_class
        if embedder is None:
            if config.emedding_method == "local":
                embedder = LocalEmbedder
            elif config.emedding_method == "openai":
                embedder = OpenAIEmbedder
            else:
                embedder = StaticEmbedder

        vector_store = self._vector_store_class
        if vector_store is None:
            if config.vector_store == "weaviate":
                from db.weaviate_store import WeaviateVectorStore
                vector_store = WeaviateVectorStore
            elif config.vector_store in ["simple", "local_file"]:
                from db.simple_store import SimpleVectorStore
                vector_store = SimpleVectorStore
            else:
                from db.weaviate_store import WeaviateVectorStore
                vector_store = WeaviateVectorStore

        return RAG(self._path , ingestor=self._ingestor, chunker_class=chunker, embedder_class=embedder, vector_store_class=vector_store)

from typing import Type
from ingester.ingester import DataSourceIngester
from chunker.chunker import FixedSizeChunker, SemanticChunker
from embedder.embedder import LocalEmbedder, OpenAIEmbedder, StaticEmbedder
from config.config import Config
from rag.rag import RAG
from models.rag import AbstractRAGPipeline


class RAGBuilder:

    def get_rag_pipeline(self , file_path : str) -> AbstractRAGPipeline:
        config = Config()
        ingestor = None
        chunker = None
        ingestor_type = file_path.split(".")[-1]
        if ingestor_type in ["pdf" , "txt" , "md"]:
            ingestor = DataSourceIngester(file_path)
        else:
            raise TypeError("")
        
        if config.chunking_method == "semantic":
            chunker = SemanticChunker
        elif config.chunking_method == "fixed":
            chunker = FixedSizeChunker
        else:
            raise TypeError("")

        if config.emedding_method == "local":
            embedder = LocalEmbedder
        elif config.emedding_method == "openai":
            embedder = OpenAIEmbedder
        else:
            embedder = StaticEmbedder

        return RAG(ingestor=ingestor, chunker_class=chunker, embedder_class=embedder)
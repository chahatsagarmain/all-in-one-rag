from typing import Type
from ingester.ingester import DataSourceIngester
from rag.rag import RAG
from models.rag import AbstractRAGPipeline


class RAGBuilder:

    def get_rag_pipeline(self , file_path : str) -> AbstractRAGPipeline:
        ingestor = None
        chunker = None
        ingestor_type = file_path.split(".")[-1]
        if ingestor_type in ["pdf" , "txt" , "md"]:
            ingestor = DataSourceIngester(file_path)
        else:
            raise TypeError("")
        return RAG(ingestor=ingestor)
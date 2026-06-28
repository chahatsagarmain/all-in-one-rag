from models.ingestor import Ingester
from models.rag import AbstractRAGPipeline

class RAG(AbstractRAGPipeline):

    def __init__(self , ingestor : Ingester):
        self.__ingestor = ingestor
    
    def build(self):
        content = self.__ingestor.ingest_data_from_path()
        print(content)
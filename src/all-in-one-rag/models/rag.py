from models.ingestor import Ingester
from abc import ABC , abstractmethod
from typing import List

class AbstractRAGPipeline(ABC):

    def __init__(self , ingestor : Ingester):
        self.ingestor = ingestor

    @abstractmethod
    def build(self):
        pass
    

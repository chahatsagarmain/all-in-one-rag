from abc import ABC ,  abstractmethod
from typing import Dict

class Embedder(ABC):

    def __init__(self , chunked_content : Dict[str , list | dict]):
        self._chunked_content = chunked_content
    
    @abstractmethod
    def create_embeddings(self) -> Dict[str, list | Dict[str, str]]:
        pass
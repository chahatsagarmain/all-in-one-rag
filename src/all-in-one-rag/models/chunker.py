from abc import ABC , abstractmethod
from models.ingestor import Content
from typing import Dict , List

class Chunker(ABC):

    def __init__(self , content : Content):
        self._content = content
    
    @abstractmethod
    def create_chunks(self) -> Dict[str , Dict[str , str] | List[str]]:
        pass




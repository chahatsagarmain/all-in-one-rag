from abc import ABC , abstractmethod
from typing import Dict
from pydantic import BaseModel

class Content(BaseModel): 
    text : str 
    metadata : Dict[str , str]

class Ingester(ABC):

    @abstractmethod
    def ingest_data_from_path(self) -> Content:
        pass

    @abstractmethod
    def _check_data_type(self) -> str:
        pass
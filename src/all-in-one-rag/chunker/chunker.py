from abc import ABC , abstractmethod

class Chunker(ABC):

    @abstractmethod
    def create_chunk(self , text : str , chunk_size : int): 
        pass


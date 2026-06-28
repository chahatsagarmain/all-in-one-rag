from abc import ABC , abstractmethod

class Ingester(ABC):

    @abstractmethod
    def ingest_data_from_path(self) -> str:
        pass

    @abstractmethod
    def _check_data_type(self) -> str:
        pass
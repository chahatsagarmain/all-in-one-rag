from models.embedder import Embedder
from openai import OpenAI
from typing import Dict
from config.model_manager import ModelManager
import os

# Suppress tqdm progress bars and HF warnings
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class LocalEmbedder(Embedder):

    def __init__(self, chunked_content: Dict[str, list | dict], model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(chunked_content)
        self._model_name = model_name
        self._model = ModelManager.get_sentence_transformer(model_name)

    def create_embeddings(self) -> Dict[str, list | Dict[str, str]]:
        chunks = self._chunked_content.get("chunks", [])
        if not chunks:
            return {"chunks": [], "embeddings": [], "metadata": self._chunked_content.get("metadata", {})}
            
        
        embeddings = self._model.encode(chunks, show_progress_bar=False )
        
        # Convert numpy array to list of list of floats
        embeddings_list = [emb.tolist() for emb in embeddings]
        return {
            "chunks": chunks,
            "embeddings": embeddings_list,
            "metadata": self._chunked_content.get("metadata", {})
        }


class OpenAIEmbedder(Embedder):

    def __init__(self, chunked_content: Dict[str, list | dict], model_name: str = "text-embedding-3-small"):
        super().__init__(chunked_content)
        self._model_name = model_name

    def create_embeddings(self) -> Dict[str, list | Dict[str, str]]:
        chunks = self._chunked_content.get("chunks", [])
        if not chunks:
            return {"chunks": [], "embeddings": [], "metadata": self._chunked_content.get("metadata", {})}
            
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        response = client.embeddings.create(
            input=chunks,
            model=self._model_name
        )
        embeddings_list = [d.embedding for d in response.data]
        return {
            "chunks": chunks,
            "embeddings": embeddings_list,
            "metadata": self._chunked_content.get("metadata", {})
        }


class StaticEmbedder(Embedder):
    _model_cache = {}

    def __init__(self, chunked_content: Dict[str, list | dict], model_name: str = "minishlab/potion-base-32M"):
        super().__init__(chunked_content)
        self._model_name = model_name
        self._model = ModelManager.get_static_model(model_name)

    def create_embeddings(self) -> Dict[str, list | Dict[str, str]]:
        chunks = self._chunked_content.get("chunks", [])
        if not chunks:
            return {"chunks": [], "embeddings": [], "metadata": self._chunked_content.get("metadata", {})}
        
        embeddings = self._model.encode(chunks)
        
        # Convert numpy array to list of list of floats
        embeddings_list = [emb.tolist() for emb in embeddings]
        return {
            "chunks": chunks,
            "embeddings": embeddings_list,
            "metadata": self._chunked_content.get("metadata", {})
        }
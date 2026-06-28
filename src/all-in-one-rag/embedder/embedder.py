from models.embedder import Embedder
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from model2vec import StaticModel
import os
import sys
import logging
import warnings
from contextlib import contextmanager
from typing import Dict

# Suppress tqdm progress bars and HF warnings
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Suppress python warnings
warnings.filterwarnings("ignore")

# Suppress logging from noisy packages
for logger_name in ["sentence_transformers", "transformers", "huggingface_hub", "model2vec"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class LocalEmbedder(Embedder):
    _model_cache = {}

    def __init__(self, chunked_content: Dict[str, list | dict], model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(chunked_content)
        self._model_name = model_name

    def create_embeddings(self) -> Dict[str, list | Dict[str, str]]:
        chunks = self._chunked_content.get("chunks", [])
        if not chunks:
            return {"chunks": [], "embeddings": [], "metadata": self._chunked_content.get("metadata", {})}
            
        if self._model_name not in LocalEmbedder._model_cache:
            with suppress_stdout_stderr():
                LocalEmbedder._model_cache[self._model_name] = SentenceTransformer(self._model_name)
        model = LocalEmbedder._model_cache[self._model_name]
        
        with suppress_stdout_stderr():
            embeddings = model.encode(chunks, show_progress_bar=False)
        
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

    def create_embeddings(self) -> Dict[str, list | Dict[str, str]]:
        chunks = self._chunked_content.get("chunks", [])
        if not chunks:
            return {"chunks": [], "embeddings": [], "metadata": self._chunked_content.get("metadata", {})}
            
        if self._model_name not in StaticEmbedder._model_cache:
            with suppress_stdout_stderr():
                StaticEmbedder._model_cache[self._model_name] = StaticModel.from_pretrained(self._model_name)
        model = StaticEmbedder._model_cache[self._model_name]
        
        with suppress_stdout_stderr():
            embeddings = model.encode(chunks)
        
        # Convert numpy array to list of list of floats
        embeddings_list = [emb.tolist() for emb in embeddings]
        return {
            "chunks": chunks,
            "embeddings": embeddings_list,
            "metadata": self._chunked_content.get("metadata", {})
        }
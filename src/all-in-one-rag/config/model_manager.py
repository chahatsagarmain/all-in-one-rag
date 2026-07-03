import os
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
from model2vec import StaticModel
from config.config import Config

class ModelManager:
    _instances: Dict[str, Any] = {}

    @staticmethod
    def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        key = f"st_{model_name}"
        if key not in ModelManager._instances:
            os.environ["TQDM_DISABLE"] = "1"
            ModelManager._instances[key] = SentenceTransformer(model_name)
        return ModelManager._instances[key]

    @staticmethod
    def get_whisper_model(model_size: str = "small") -> WhisperModel:
        key = f"whisper_{model_size}"
        if key not in ModelManager._instances:
            config = Config()
            ModelManager._instances[key] = WhisperModel(
                model_size_or_path=model_size,
                device=config.whisper_device,
                compute_type="int8",
            )
        return ModelManager._instances[key]

    @staticmethod
    def get_static_model(model_name: str = "minishlab/potion-base-32M") -> StaticModel:
        key = f"static_{model_name}"
        if key not in ModelManager._instances:
            ModelManager._instances[key] = StaticModel.from_pretrained(model_name)
        return ModelManager._instances[key]

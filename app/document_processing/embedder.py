import os
from typing import List

import numpy as np

from app.engine.config import DocConfig
from app.utils.logger import get_logger
from openai import OpenAI

logger = get_logger(__name__)

RAG_EMBEDDING_API_KEY_ENVIRON = "RAG_EMBEDDING_API_KEY"


class Embedder:
    def __init__(self, config: DocConfig):
        self.model_name = config.embedding_model
        self.model_path = config.embedding_model_path

        api_key = os.environ.get(RAG_EMBEDDING_API_KEY_ENVIRON, None)
        if api_key is None:
            api_key = "api_key" if config.emb_api_key == "" else config.emb_api_key
        self.client = OpenAI(base_url=config.emb_api_url, api_key=api_key)

    def embed(self, text: List[str]) -> List[List[float]]:
        """calculate embedding of input text

        Args:
            text (List[str]): list of input text

        Returns:
            List: list of embeddings
        """
        response = self.client.embeddings.create(
            model="model",
            input=text,
        )
        return np.array([item.embedding for item in response.data], dtype=np.float32)

import os
from typing import List

from engine.config import EmbeddingConfig
from utils.logger import get_logger

from FlagEmbedding import FlagAutoModel

logger = get_logger(__name__)


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.model_name = config.model
        self.model_path = config.path
        if os.path.exists(self.model_path):
            logger.info(f"Offline loading, model path is {self.model_path}")
            self.model = FlagAutoModel.from_finetuned(
                self.model_path,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )
        else:
            logger.info(f"Online loading, model name is {self.model_name}")
            self.model = FlagAutoModel.from_finetuned(
                self.model_name,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )

    def embed(self, text: List[str]) -> List:
        embeddings = self.model.encode(text)
        return embeddings.tolist()

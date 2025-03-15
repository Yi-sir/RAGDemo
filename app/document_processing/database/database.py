from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from engine.config import DocConfig
from utils.logger import get_logger
from RAG_demo.app.document_processing.database.database_faiss import DataBaseFaiss

logger = get_logger(__name__)

VECTORSTORECLSMAP = {
    "Faiss": DataBaseFaiss
}

class DataBase(ABC):
    def __init__(self, config: DocConfig):
        self.dimension = config.dimension
        self.topk = config.topk
        self.index_map = {}
    
    @abstractmethod
    def add_vector(self, filename: str, vectors: np.array):
        """add vector to faiss database

        Args:
            filename (str): filename
            vectors (np.array): all embeddings, shape like [num of vectors, dimension]

        Raises:
            ValueError: dimension mismatch
        """
        raise NotImplementedError("add_vector must be implemented in subclasses.")
    
    @abstractmethod
    def search(self, query: np.array, k: int = 5) -> List[Tuple[str, int, float]]:
        """search for top k vectors with max similarity
        
        Args:
            query_vector: query vector

        Returns:
            List[Tuple[str, int, float]]: a list of (filename, chunk id, similarity)
        """
        raise NotImplementedError("search must be implemented in subclasses.")
    
    @classmethod
    def from_config(cls, config: DocConfig):
        vector_cls = VECTORSTORECLSMAP.get(config.database_method, None)
        if vector_cls == None:
            raise NameError(f"No such method! Method name: {config.database_method}")
        return vector_cls(config)
    
    def remove_vectors(self, filename: str):
        """remove vectors by filename

        Args:
            filename (str): filename
        """
        self.index_map.pop(filename)
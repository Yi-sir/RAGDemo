from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Dict
import numpy as np
from app.engine.config import DocConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataBase(ABC):
    
    _subclasses: Dict[str, Type["DataBase"]] = {}
    
    def __init__(self, config: DocConfig):
        self.dimension = config.dimension
        self.topk = config.topk
        self.index_map = {}

    @classmethod
    def register_subclass(cls, class_name, subclass: Type["DataBase"]):
        """register a subclass"""
        cls._subclasses[class_name] = subclass
    
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
        """create subclass instance from config

        Args:
            config (DocConfig): config

        Raises:
            ValueError: if invalid subclass name

        Returns:
            _type_: subclass instance
        """
        cls_type = config.database_method
        if cls_type not in cls._subclasses:
            raise ValueError(f"Unknown database type: {cls_type}")
        return cls._subclasses[cls_type](config)
    
    def remove_vectors(self, filename: str):
        """remove vectors by filename

        Args:
            filename (str): filename
        """
        self.index_map.pop(filename)
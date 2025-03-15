from abc import ABC, abstractmethod
from typing import List, Dict, Type

from app.engine.config import DocConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocSplitterBase(ABC):
    
    _subclasses: Dict[str, Type["DocSplitterBase"]] = {}
    
    def __init__(self, config: DocConfig):
        self.method = config.split_method
        logger.info(f"Document split method: {self.method}")
        
    @classmethod
    def register_subclass(cls, class_name, subclass: Type["DocSplitterBase"]):
        """register a subclass"""
        cls._subclasses[class_name] = subclass

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError("split_text must be implemented in subclasses.")

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
        splitter_type = config.split_method
        if splitter_type not in cls._subclasses:
            raise ValueError(f"Unknown generator type: {splitter_type}")
        return cls._subclasses[splitter_type](config)

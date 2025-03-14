from typing import List
from abc import ABC, abstractmethod
from engine.config import DocConfig
from utils.logger import get_logger
from splitter.fixed_len_splitter import FixedLengthSplitter

logger = get_logger(__name__)

SPLITTERCLSMAP = {
    "FixedLength": FixedLengthSplitter
}

class DocSplitterBase(ABC):
    def __init__(self, config: DocConfig):
        self.method = config.split_method
        logger.info(f"Document split method: {self.method}")
    
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError("split_text must be implemented in subclasses.")

    @classmethod
    def from_config(cls, config):
        splitter_cls = SPLITTERCLSMAP.get(config.split_method, None)
        if splitter_cls == None:
            raise NameError(f"No such method! Method name: {config.split_method}")
        return splitter_cls(config)
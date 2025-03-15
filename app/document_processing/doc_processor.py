import re
from pathlib import Path
from typing import List, Union, Dict, Tuple

import numpy as np
import pdfplumber  # for PDF
from docx import Document  # for DOCX
from app.document_processing.embedder import Embedder
from app.engine.config import DocConfig
from app.utils.logger import get_logger

from app.document_processing.database.database import DataBase
from app.document_processing.splitter.doc_splitter import \
    DocSplitterBase

logger = get_logger(__name__)


def load_document(file_path: Union[str, Path]) -> str:
    """load and extract file
    
    Args:
        file_path: file path

    Raises:
        FileNotFoundError: invalid path
        ValueError: invalid file format

    Returns:
        str: texts in the file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if file_path.suffix == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
    elif file_path.suffix == ".docx":
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif file_path.suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return text


def clean_text(text: str) -> str:
    """clean text
    
    Args:
        text: input text

    Returns:
        str: cleaned text
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text


class DocProcessor:
    """document processor"""

    def __init__(self, config: DocConfig):
        # Dict[filename, Dict[chunk id, text]]
        self.doc_chunk_map: Dict[str, Dict[int, str]] = {}
        self.embedder = Embedder(config)
        self.splitter = DocSplitterBase.from_config(config)
        self.vector_store = DataBase.from_config(config)
        
    def _update_doc_map(self, file_path: str, chunks: List[str]):
        """record each chunk with id and filename

        Args:
            file_path (str): filename
            chunks (List[str]): chunks, splitted by splitter
        """
        self.doc_chunk_map[file_path] = {index: text for index, text in enumerate(chunks)}
        
        
    def _get_chunk_by_name_and_id(self, file_name: str, id: int) -> str:
        """get a text chunk from map

        Args:
            file_name (str): file name
            id (int): chunk id
        """
        if not file_name in self.doc_chunk_map.keys():
            logger.error(f"Cannot find a file named {file_name} in dict.")
            raise FileNotFoundError()
        if not id in self.doc_chunk_map[file_name].keys():
            logger.error(f"Invalid chunk id, filename: {file_name}, chunk_id: {id}")
            raise IndexError()
        return self.doc_chunk_map[file_name][id]

    def process_document(self, file_path: str):
        """process document and add to database
        
        Args:
            file_path (str): file path
        """
        if file_path in self.doc_chunk_map.keys():
            logger.error(f"Found file with the same name {file_path}")
            raise FileExistsError()
        logger.info(f"Process new file {file_path}")
        text = load_document(file_path)
        text = clean_text(text)
        chunks = self.splitter.split_text(text)
        self._update_doc_map(file_path, chunks)
        vectors = self.embedder.embed(chunks)
        self.vector_store.add_vector(file_path, vectors)
        
    def remove_document(self, file_path: str):
        """remove a file

        Args:
            file_path (str): filename
        """
        if not file_path in self.doc_chunk_map.keys():
            logger.warning(f"Cannot found file with the same name {file_path}")
            raise ValueError()
        self.vector_store.remove_vectors(file_path)
        self.doc_chunk_map.pop(file_path)
        logger.info(f"File {file_path} is removed")
        

    def search_ralated_chunk(self, text: str) -> List[Tuple[str, str]]:
        """search related chunks with input text

        Args:
            text (str): input text

        Returns:
            List[Tuple[str, str]]: List[Tuple(filename, chunk)]
        """
        embedding = self.embedder.embed(text)
        res = self.vector_store.search(embedding) # (filename, chunk id, similarity)
        ret = []
        for (filename, chunk_id, _) in res:
            ret.append((filename, self._get_chunk_by_name_and_id(filename, chunk_id)))
        return ret        

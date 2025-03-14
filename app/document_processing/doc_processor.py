import re
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np
import pdfplumber  # 解析PDF
from docx import Document  # 解析DOCX
from embedder import Embedder
from doc_splitter import DocSplitterBase
from sentence_transformers import SentenceTransformer

from engine.config import DocConfig


def load_document(file_path: Union[str, Path]) -> str:
    """
    load and extract file
    :param file_path: file path
    :return: content of file
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
    """
    clean text
    :param text: input text
    :return: cleaned text
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text

class VectorStore:
    """向量存储工具"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # 使用L2距离的FAISS索引

    def add_vectors(self, vectors: List[List[float]]):
        """
        添加向量到索引
        :param vectors: 向量列表
        """
        self.index.add(np.array(vectors))

    def search(self, query_vector: List[float], k: int = 5) -> List[int]:
        """
        搜索最相似的k个向量
        :param query_vector: 查询向量
        :param k: 返回的结果数量
        :return: 最相似的向量的索引列表
        """
        distances, indices = self.index.search(np.array([query_vector]), k)
        return indices[0].tolist()


class DocProcessor:
    """document processor"""

    def __init__(self, config: DocConfig):
        self.embedder = Embedder(config)
        self.splitter = DocSplitterBase.from_config(DocConfig)
        # self.vector_store = vector_store

    def process_document(self, file_path: str):
        """
        process document and add to database
        :param file_path: file path
        """
        text = load_document(file_path)
        text = clean_text(text)
        chunks = self.splitter.split_text(text)
        vectors = self.embedder.embed(chunks)
        self.vector_store.add_vectors(vectors)

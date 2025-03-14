from config import RAGConfig
from models.generator.generator import Generator
from retrieval.retriever import Retriever
from utils.logger import get_logger

# TODO: 把embedding改成doc processor的模块，这里import doc processor
from RAG_demo.app.document_processing.doc_processor import DocProcessor

logger = get_logger(__name__)


class RAGEngine:
    def __init__(self, config: RAGConfig):
        self.generator = Generator.from_config(config.llm_config)
        self.retriever = Retriever.from_config(config.retrieval_config)
        self.doc_processor = DocProcessor.from_config(config.emb_config)

    def run():
        pass

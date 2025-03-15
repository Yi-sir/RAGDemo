from typing import Dict

from app.engine.config import RAGConfig
from app.models.generator.generator import Generator
from app.utils.logger import get_logger

from app.document_processing.doc_processor import DocProcessor

logger = get_logger(__name__)


class RAGEngine:
    def __init__(self, config: RAGConfig):
        self.generator = Generator.from_config(config.llm_config)
        self.doc_processor = DocProcessor(config.doc_config)

    def add_doc(self, file_path: str) -> bool:
        """add a document to rag system

        Args:
            file_path (str): file path
            
        Returns:
            whether is successfully added
        """
        try:
            self.doc_processor.process_document(file_path=file_path)
            logger.info(f"Document added successfully: {file_path}")
            return True
        except:
            logger.error(f"Failed to add document: {file_path}")
            return False
        
    def remove_doc(self, file_path: str) -> bool:
        """remove a document from rag system

        Args:
            file_path (str): file path

        Returns:
            bool: whether is successfully removed
        """
        try:
            self.doc_processor.remove_document(file_path)
            logger.info(f"Document removed successfully: {file_path}")
            return True
        except:
            logger.error(f"Failed to remove document: {file_path}")
            return False
        
    def query(self, question: str) -> Dict:
        """generate answer to the question from user

        Args:
            question (str): user question
        """
        try:
            results = self.doc_processor.search_ralated_chunk(question)
            context = ["\n".join(text) for _, text in results][0]
            
            # 这个接口是不是做成generate(context, question) ?
            # 还有对话历史
            answer = self.generator.generate(context + results)
            return {
                "answer": answer,
                "reference": results
            }
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "answer": None,
                "reference": None
            }

from typing import Dict, List
import os

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
        # TODO: 是不是要支持一下List提交？
        logger.info(f"Add document: {file_path}")
        try:
            self.doc_processor.process_document(file_path)
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
        real_path = os.path.abspath(file_path)
        try:
            self.doc_processor.remove_document(real_path)
            logger.info(f"Document removed successfully: {real_path}")
            return True
        except:
            logger.error(f"Failed to remove document: {real_path}")
            return False
        
    def query(self, question: str) -> Dict:
        """generate answer to the question from user

        Args:
            question (str): user question
        """
        try:
            results = self.doc_processor.search_ralated_chunk(question)
            context = [tup[1] for tup in results]
            context = "\n".join(context)
            # 这个接口是不是做成generate(context, question) ?
            # 还有对话历史
            answer = self.generator.generate(context + question)
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

    def get_status(self) -> str:
        """get status of service

        Returns:
            bool: True if alive, else False
        """
        
        return {"status": "alive"}
    
    def get_doc_list(self) -> List[str]:
        """get all documents stored in database

        Returns:
            List[str]: list of document name
        """
        return self.doc_processor.get_doc_list()
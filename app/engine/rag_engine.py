import os
from typing import Dict, List, Union, Optional, Tuple

from app.document_processing.doc_processor import (DocProcessor,
                                                   check_if_support_docx)
from app.engine.config import RAGConfig
from app.models.generator.generator import Generator
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGEngine:
    def __init__(self, config: RAGConfig):
        self.generator = Generator.from_config(config.llm_config)
        self.doc_processor = DocProcessor(config.doc_config)
        logger.info(f"RAGEngine is initialized with config {config}")
        self.chat_history = []

    def _add_single_file(self, file_path: str) -> bool:
        """add a document to rag system

        Args:
            file_path (str): file path

        Returns:
            whether is successfully added
        """
        real_path = os.path.abspath(file_path)
        logger.info(f"Add document: {real_path}")
        try:
            self.doc_processor.process_document(real_path)
            logger.info(f"Document added successfully: {real_path}")
            return True
        except:
            logger.error(f"Failed to add document: {real_path}")
            return False

    def add_doc(self, file_path: Union[str, List[str]]) -> bool:
        """add a document or a list of documents to rag system

        Args:
            file_path (Union[str, List[str]]): file path(s)

        Returns:
            whether is successfully added
        """
        if isinstance(file_path, str):
            return self._add_single_file(file_path)
        elif isinstance(file_path, List[str]):
            for file in file_path:
                ret = self._add_single_file(file)
                if not ret:
                    return False
            return True

    def remove_doc(self, file_path: str) -> bool:
        """remove a document from rag system

        Args:
            file_path (str): file path

        Returns:
            bool: whether is successfully removed
        """
        real_path = os.path.abspath(file_path)
        logger.info(f"Remove document: {real_path}")
        try:
            self.doc_processor.remove_document(real_path)
            logger.info(f"Document removed successfully: {real_path}")
            return True
        except:
            logger.error(f"Failed to remove document: {real_path}")
            return False

    def query(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """generate answer to the question from user

        Args:
            question (str): user question
        """
        try:
            results = self.doc_processor.search_ralated_chunk(question)
            prompt = self._make_prompt(question, results, history)
            # 这个接口是不是做成generate(context, question) ?
            # 还有对话历史
            answer = self.generator.generate(prompt)
            self.chat_history.append((question, answer))
            return {"answer": answer, "reference": results}
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {"answer": None, "reference": None}

    def check_query_stream_support(self) -> bool:
        """check if llm backend supports generate stream

        Returns:
            bool: support or not
        """
        return self.generator.check_query_stream_support()
    
    def _make_prompt(self, question: str, search_results: List[Tuple[str, str]], history: Optional[List[Tuple[str, str]]]=None):
        """prompt maker

        Args:
            question (str): user input
            search_results (List[Tuple[str, str]]): related chunks in uploaded files
            history (Optional[List[Tuple[str, str]]], optional): chat history. Defaults to None.

        Returns:
            _type_: prompt for llm input
        """
        context = [tup[1] for tup in search_results]
        context = "\n".join(context)
        prompt = context + "\n以上是检索到的参考文本\n"
        if history:
            prompt += "\n".join(f"Q:{q}\nA:{a}\n" for q, a in history)
            prompt += "除了参考文本外，以上是历史对话。其中'Q:'后面的是问题，'A:'后面的是答案\n"
        prompt += f"请根据你的知识和检索结果回答以下问题\nQ:{question}"
        return prompt

    def query_stream(self, question: str, history: Optional[List[Tuple[str, str]]] = None):
        """generate answer to question from user, in stream mode

        Args:
            question (str): user question
        """
        try:
            results = self.doc_processor.search_ralated_chunk(question)
            prompt = self._make_prompt(question, results, history)
            stream = self.generator.generate_stream(prompt)
            complete_answer = ""
            for chunk in stream:
                if len(chunk.choices) == 0:
                    continue
                partial_answer = chunk.choices[0].delta.content
                if partial_answer is not None:
                    complete_answer += partial_answer
                    yield {"answer": partial_answer, "reference": results}
            self.chat_history.append((question, complete_answer))
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            yield {"answer": None, "reference": None}

    def _get_history(self):
        """get chat history
        """
        # TODO: 考虑最大长度？
        return self.chat_history

    def query_chat(self, question: str):
        history = self._get_history()
        return self.query(question, history)
    
    def query_chat_stream(self, question: str):
        history = self._get_history()
        return self.query_stream(question, history)

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

    def update_topk(self, topk: int):
        """update topk param while retrieval

        Args:
            topk (int): _description_
        """
        self.doc_processor.update_topk(topk)

    def check_if_support_docx(self) -> bool:
        return check_if_support_docx()

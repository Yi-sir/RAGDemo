import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(BASE_DIR, "app/config")
RAG_ENGINE_CONFIG_PATH = os.path.join(JSON_DIR, "config.json")
RAG_LOGGER_CONFIG_PATH = os.path.join(JSON_DIR, "logging.json")

sys.path.append(BASE_DIR)
from app.utils.logger import get_logger, setup_logging

setup_logging(RAG_LOGGER_CONFIG_PATH)

logger = get_logger(__name__)

from app.engine.config import RAGConfig
from app.engine.rag_engine import RAGEngine

if __name__ == "__main__":

    engine_config = RAGConfig.from_json(RAG_ENGINE_CONFIG_PATH)
    engine = RAGEngine(engine_config)

    logger.info("Test add document")
    engine.add_doc("./rag_test.txt")

    logger.info("Test query")
    results = engine.query("介绍一下今天、明天、后天和大后天的天气")
    print(results)

    if engine.check_query_stream_support():
        logger.info("Test query stream")
        stream = engine.query_stream("介绍一下今天、明天、后天和大后天的天气")
        for partial_result in stream:
            if partial_result["answer"] is not None:
                print(partial_result["answer"])

    logger.info("Test remove document")
    engine.remove_doc("./rag_test.txt")

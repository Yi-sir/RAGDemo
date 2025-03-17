import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(BASE_DIR, "app/config")
RAG_ENGINE_CONFIG_PATH = os.path.join(JSON_DIR, "config.json")
RAG_LOGGER_CONFIG_PATH = os.path.join(JSON_DIR, "logging.json")

sys.path.append(BASE_DIR)
from app.utils.logger import setup_logging
setup_logging(RAG_LOGGER_CONFIG_PATH)

from app.engine.rag_engine import RAGEngine
from app.engine.config import RAGConfig


if __name__ == "__main__":

    engine_config = RAGConfig.from_json(RAG_ENGINE_CONFIG_PATH)
    engine = RAGEngine(engine_config)
    
    engine.add_doc("./rag_test.txt")
    
    results = engine.query("介绍一下今天、明天、后天和大后天的天气")
    print(results)
    
    engine.remove_doc("./rag_test.txt")

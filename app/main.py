import os

from engine import RAGConfig, RAGEngine
from utils.logger import setup_logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(BASE_DIR, "config")
RAG_ENGINE_CONFIG_PATH = os.path.join(JSON_DIR, "config.json")
RAG_LOGGER_CONFIG_PATH = os.path.join(JSON_DIR, "logging.json")

if __name__ == "__main__":
    setup_logging(RAG_LOGGER_CONFIG_PATH)

    engine_config = RAGConfig.from_json(RAG_ENGINE_CONFIG_PATH)
    engine = RAGEngine(engine_config)

    engine.run()

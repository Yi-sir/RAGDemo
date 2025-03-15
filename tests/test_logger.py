import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(BASE_DIR, "app/config")
RAG_ENGINE_CONFIG_PATH = os.path.join(JSON_DIR, "config.json")
RAG_LOGGER_CONFIG_PATH = os.path.join(JSON_DIR, "logging.json")

sys.path.append(BASE_DIR)

from app.utils.logger import setup_logging, get_logger

if __name__ == "__main__":
    setup_logging(RAG_LOGGER_CONFIG_PATH)
    logger = get_logger(__name__)
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.critical("critical")
    logger.error("error")

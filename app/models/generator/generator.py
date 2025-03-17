from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Dict
import importlib
import os
import sys
from app.engine.config import GeneratorConfig

GENERATOR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(GENERATOR_DIR)

GENERATOR_CONFIG_MODULENAME_CLASSNAME_MAP = {
    "api": ("generator_api", "GeneratorApi"),
    "local": ("generator_local", "GeneratorLocal")
}


class Generator(ABC):
    
    def __init__(self, config: GeneratorConfig = None):
        """
        init generator
        :param model_name: such as "gpt-3.5-turbo" or "llama-2-13b"
        :param config: GeneratorConfig from Engine
        """
        self.model_name = config.model
        self.config = config
        # self._validate_config(self.config)

    def _validate_config(self, config):
        if "temperature" in config and not 0 <= config["temperature"] <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if "max_tokens" in config and config["max_tokens"] <= 0:
            raise ValueError("Max tokens must be a positive integer")
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Call llm for generation
        :param prompt: input prompt
        :param kwargs: other params, such as temperature and max_tokens
        :return: output text
        """
        raise NotImplementedError("generate must be implemented in subclasses.")

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    @classmethod
    def from_config(cls, config: GeneratorConfig):
        """create subclass instance from config

        Args:
            config (GeneratorConfig): config

        Raises:
            ValueError: if invalid subclass name

        Returns:
            _type_: subclass instance
        """
        generator_type = config.backend_type.lower()
        if generator_type not in GENERATOR_CONFIG_MODULENAME_CLASSNAME_MAP:
            raise ValueError(f"Invalid generator backend type: {generator_type}")
        module_name, class_name = GENERATOR_CONFIG_MODULENAME_CLASSNAME_MAP[generator_type]
        module = importlib.import_module(module_name)
        derived_class = getattr(module, class_name, None)
        if not derived_class or not issubclass(derived_class, cls):
            raise ValueError(f"Class {class_name} not found or is not subclass of {cls}")
        return derived_class(config)
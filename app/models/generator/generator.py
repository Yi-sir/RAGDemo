from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Dict

from app.engine.config import GeneratorConfig


class Generator(ABC):
    
    _subclasses: Dict[str, Type["Generator"]] = {}
    
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
        
        
    @classmethod
    def register_subclass(cls, class_name):
        """register a subclass"""
        def decorator(subclass: Type["Generator"]):
            cls._subclasses[class_name] = subclass
            return subclass
        return decorator

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
        generator_type = config.backend_type
        if generator_type not in cls._subclasses:
            raise ValueError(f"Unknown generator type: {generator_type}")
        return cls._subclasses[generator_type](config)
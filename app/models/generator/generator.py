from abc import ABC, abstractmethod
from typing import Any, Optional

from engine.config import GeneratorConfig
from generator_api import GeneratorAPI
from generator_local import GeneratorLocal


class Generator(ABC):
    def __init__(self, config: Optional[GeneratorConfig[str, Any]] = None):
        """
        init generator
        :param model_name: such as "gpt-3.5-turbo" or "llama-2-13b"
        :param config: GeneratorConfig from Engine
        """
        self.model_name = config.model
        self.config = config
        self._validate_config(self.config)

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
        if config.api_key is None or config.api_key == "":
            generator_cls = GeneratorLocal
        elif config.path is None or config.path == "":
            generator_cls = GeneratorAPI

        return generator_cls(config)

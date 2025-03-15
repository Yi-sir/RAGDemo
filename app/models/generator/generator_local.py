import importlib
from typing import Any, Dict

import torch
from app.models.generator.generator import Generator, GeneratorConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)

@Generator.register_subclass("Local")
class GeneratorLocal(Generator):
    def __init__(self, config: GeneratorConfig = None):
        super().__init__(config)
        try:
            module = importlib.import_module("torch_tpu")
            logger.info("found torch_tpu")
            self.device = "tpu" if torch.tpu.is_available() else "cpu"
        except ModuleNotFoundError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )

    def make_generation_config(self, **kwargs) -> Dict[str, Any]:
        """
        merge default params and user inputs
        :param kwargs: user inputs
        :return: merged config
        """
        default_config = {
            "temperature": 1.0,
            "max_tokens": 512,
            "top_p": 1.0,
            "top_k": 50,
        }

        # priority: kwargs > self.config > default_config
        generation_config = {**default_config, **self.config, **kwargs}

        self._validate_config(generation_config)

        return generation_config

    def generate(self, prompt: str, **kwargs) -> str:
        generation_config = self.make_generation_config(**kwargs)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=generation_config.get("max_tokens", 512),
                temperature=generation_config.get("temperature", 1.0),
                **generation_config
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

Generator.register_subclass("LOCAL", GeneratorLocal)
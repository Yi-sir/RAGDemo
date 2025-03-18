import importlib
from typing import Any, Dict

import torch
from app.models.generator.generator import Generator, GeneratorConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def is_module_available(module_name: str):
    try:
        module = importlib.import_module(module_name, None)
        logger.info(f"{module_name} is available.")
        return True
    except ModuleNotFoundError:
        return False


class GeneratorLocal(Generator):
    def __init__(self, config: GeneratorConfig = None):
        super().__init__(config)
        if is_module_available("vllm"):
            self.backend = "vllm"
            self._init_vllm()
        elif is_module_available("transformers"):
            self.backend = "transformers"
            self._init_transformers()
        else:
            logger.info(
                "No available local generation framework, please use GeneratorApi"
            )
        self._stream_support = False

    def _init_vllm(self):
        """init vllm backend if vllm is available

        Args:
            config (GeneratorConfig): generator config
        """
        logger.info("Using vllm as local generation backend")
        from vllm import LLM, SamplingParams

        self.vllm_sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        self.model = LLM(model=self.model_path)

    def _init_transformers(self):
        """init transformers if transformers is available

        Args:
            config (GeneratorConfig): generator config
        """
        logger.info("Using transformers as local generation backend")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            module = importlib.import_module("torch_tpu")
            logger.info("found torch_tpu")
            self.device = "tpu" if torch.tpu.is_available() else "cpu"
        except ModuleNotFoundError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(
            self.device
        )

    def _generate_vllm(self, prompt: str, **kwargs) -> str:
        outputs = self.model.generate(prompt, self.vllm_sampling_params)
        return outputs.output[0].text

    def _generate_transformers(self, prompt: str, **kwargs) -> str:
        generation_config = self.make_generation_config(**kwargs)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=generation_config.get("max_tokens", 512),
                temperature=generation_config.get("temperature", 1.0),
                **generation_config,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def make_generation_config(self, **kwargs) -> Dict[str, Any]:
        """merge default params and user inputs

        Args:
            kwargs: user inputs

        Returns:
            merged config
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
        """generate answer with prompt and kwargs

        Args:
            prompt (str): input question

        Returns:
            str: answer
        """
        if self.backend == "vllm":
            return self._generate_vllm(prompt, kwargs)
        elif self.backend == "transformers":
            return self._generate_transformers(prompt, kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        """generate answer with prompt and kwargs in stream

        Args:
            prompt (str): input question

        Returns:
            generator
        """
        # if self.backend == "vllm":
        #     return self._generate_vllm(prompt, kwargs)
        # elif self.backend == "transformers":
        #     return self._generate_transformers(prompt, kwargs)
        pass


Generator.register_subclass("LOCAL", GeneratorLocal)

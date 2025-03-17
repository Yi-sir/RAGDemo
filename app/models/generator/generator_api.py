import openai
from app.models.generator.generator import Generator, GeneratorConfig
from app.utils.logger import get_logger
import os

logger = get_logger(__name__)

RAG_GENERATOR_API_KEY_ENVIRON = "RAG_GENERATOR_API_KEY"

class GeneratorApi(Generator):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        api_key = os.environ.get(RAG_GENERATOR_API_KEY_ENVIRON, None)
        if api_key is None:
            api_key = config.api_key
        self.client = openai.OpenAI(api_key=api_key, base_url=config.api_url)
        logger.info(f"GeneratorAPI initialized, api_url: {config.api_url}")

    def generate(self, prompt: str, **kwargs) -> str:
        # generation_config = {**self.config, **kwargs}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是LLM智能助手"
                },
                {
                    "role": "user",
                    "content": prompt   
                }
            ]
        )

        # 返回生成的文本
        return response.choices[0].message.content

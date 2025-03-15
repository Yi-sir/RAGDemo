import openai
from app.models.generator.generator import Generator, GeneratorConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)

@Generator.register_subclass("Api")
class GeneratorAPI(Generator):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config.api_key, base_url=config.api_url)
        logger.info(f"GeneratorAPI initialized, api_key: {config.api_key}, api_url: {config.api_url}")

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

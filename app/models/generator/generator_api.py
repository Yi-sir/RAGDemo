import openai
from app.models.generator.generator import Generator, GeneratorConfig

@Generator.register_subclass("Api")
class GeneratorAPI(Generator):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        self.api_key = config.api_key
        openai.api_key = self.api_key

    def generate(self, prompt: str, **kwargs) -> str:
        generation_config = {**self.config, **kwargs}

        response = openai.Completion.create(
            model=self.model_name, message=[
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

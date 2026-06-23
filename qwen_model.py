# qwen_model.py
import openai
from deepeval.models.base_model import DeepEvalBaseLLM

class Qwen(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "qwen-max", api_key: str = None):
        # 1. 初始化模型名称和API密钥
        self.model_name = model_name
        self.api_key = api_key
        # 2. 配置OpenAI客户端，使其指向阿里云百炼平台的兼容端点[reference:3]
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def load_model(self):
        # 返回模型对象，对于API调用，通常返回客户端实例本身[reference:4]
        return self.client

    def generate(self, prompt: str) -> str:
        # 核心的同步生成方法[reference:5]
        chat_model = self.load_model()
        try:
            response = chat_model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                # 可以在这里调整温度等参数
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        # 异步生成方法，可以复用同步方法[reference:6]
        return self.generate(prompt)

    def get_model_name(self) -> str:
        # 返回模型名称[reference:7]
        return self.model_name
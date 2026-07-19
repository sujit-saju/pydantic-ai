from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
import os

load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

class ResponseModel(BaseModel):
    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")

model = OllamaModel(
    model_name="qwen2.5:7b",
    provider=OllamaProvider(
        base_url=ollama_base_url,   # <-- NOTE THE /v1
    ),
)

agent = Agent(
    model=model,
    output_type=ResponseModel,
    system_prompt="You are an intelligent customer support assistant.",
)

result = agent.run_sync(
    "How can I track my Amazon order #1234?"
)

print(result.output)
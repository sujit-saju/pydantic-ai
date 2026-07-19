from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

agent = Agent(
    OllamaModel(
        model_name="llama3.2:1b",
        provider=OllamaProvider(
            base_url="http://127.0.0.1:11434/v1",
        ),
    )
)

result = agent.run_sync("Say hello in one sentence.")

print(result.output)
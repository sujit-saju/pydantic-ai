import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

load_dotenv()

geminiModel = os.getenv("GEMINI_API_KEY")

model = GoogleModel("gemini-2.5-pro")

# --------------------------------------------------------------
# 1. Simple Agent - Hello World Example
# --------------------------------------------------------------

# agent = Agent(
#     "google:gemini-3.5-flash",
#     instructions="You are a helpful assistant.",
# )

# result = agent.run_sync("How can I track my order #12345. I used Amazon.in")
# print(result.output)

# response2 = agent.run_sync(
#     user_prompt="What was my previous question?",
#     message_history=result.new_messages(),
# )

# print(response2.output)

# --------------------------------------------------------------
# 2. Agent with Structured Response
# --------------------------------------------------------------


class ResponseModel(BaseModel):
    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")


agent1 = Agent(
    model=model,
    output_type=ResponseModel,
    system_prompt=(
        "You are an intelligent customer support",
        "Analyze queries carefully and provide structured responses.",
    ),
)

response = agent1.run_sync("How can I track my amazon order #1234")
print(response.data.model_dump_json(indent=2))

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from utils.markdown import to_markdown

load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
print(os.getenv("OLLAMA_BASE_URL"))


class ResponseModel(BaseModel):
    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")


class Order(BaseModel):
    order_id: str
    status: str
    items: list[str]


# CUSTOMER SCHEMA
class CustomerDetails(BaseModel):
    customer_id: str
    name: str
    email: str
    orders: Optional[list[Order]] = None


agent = Agent(
    model=OllamaModel(
        model_name="llama3.2:1b",
        provider=OllamaProvider(
            base_url=ollama_base_url,
        ),
        format="json"
    ),
    output_type=NativeOutput(ResponseModel),
    deps_type=CustomerDetails,
    retries=3,
    system_prompt=(
        "You are a customer support assistant. "
        "Answer using the customer information provided. "
        "Return ONLY valid JSON matching the ResponseModel schema. "
        "Do not return markdown. "
        "Do not create tools or function definitions."
    ),
)


@agent.system_prompt
async def add_customer_name(ctx: RunContext[CustomerDetails]) -> str:
    return (
        f"Customer name is {to_markdown(ctx.deps)}. Please provide a helpful response."
    )


customer = CustomerDetails(
    customer_id="12345",
    name="John Doe",
    email="john.doe@example.com",
    orders=[
        Order(order_id="1234", status="Shipped", items=["Item A", "Item B"]),
    ],
)

response = agent.run_sync("What did I order and when will it arrive?", deps=customer)
response.all_messages()

print(response.output.model_dump_json(indent=2))

import os
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, NativeOutput, ModelSettings  # <-- Added ModelSettings
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

class ResponseModel(BaseModel):
    response: str
    needs_escalation: bool
    follow_up_required: bool
    sentiment: str = Field(description="Customer sentiment analysis")

class Order(BaseModel):
    order_id: str
    status: str
    items: list[str]

class CustomerDetails(BaseModel):
    customer_id: str
    name: str
    email: str
    orders: Optional[list[Order]] = None

shipping_info_db: Dict[str ,str] = {
    "#12345": "Shipped on 2026-01-15, expected delivery on 2026-01-20",
    "#67890": "Processing, expected delivery on 2026-01-25",
}

customer = CustomerDetails(
    customer_id="12345",
    name="John Doe",
    email="john.doe@gmail.com"
)

agent = Agent(
    model=OllamaModel(
        model_name="qwen2.5:7b",
        provider=OllamaProvider(
            base_url=ollama_base_url,
        ),
    ),
    output_type=NativeOutput(ResponseModel),
    deps_type=CustomerDetails,
    retries=3,
    model_settings=ModelSettings(temperature=0.0),  # <-- Forces strict, logical behavior
    system_prompt=(
        "You are a customer support assistant. "
        "When a user asks about an order or shipping status, you MUST use the `get_shipping_status` tool first. "
        "Do not guess or hallucinate tracking information. Use the exact tool output to write your final response."
    ),
)

@agent.tool_plain()
def get_shipping_status(order_id: str) -> str:
    """Get the shipping status for a given order ID."""
    # Robustness: auto-prepends '#' if the model passes a raw numeric string like '12345'
    if not order_id.startswith("#"):
        order_id = f"#{order_id}"
        
    shipping_status = shipping_info_db.get(order_id)
    if shipping_status is None:
        # This will raise a ModelRetry, forcing the model to fix its input and try again
        raise ModelRetry(
            f"No shipping information found for order ID {order_id}. "
            "Please check if the order number matches our records (#12345 or #67890) and try again."
        )
    return shipping_status

# Changed prompt to 12345 to successfully hit the database, or keep 18345 to watch it trigger ModelRetry!
response = agent.run_sync(
    user_prompt="What's the status of my last order 11111?", deps=customer
)

print(response.output.model_dump_json(indent=2))
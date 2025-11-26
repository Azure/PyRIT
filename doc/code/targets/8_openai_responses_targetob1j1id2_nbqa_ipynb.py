# %%NBQA-CELL-SEP52c935
from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

target = OpenAIResponseTarget()
# For an AzureOpenAI endpoint with Entra ID authentication enabled, use the following command instead. Make sure to run `az login` first.
# target = OpenAIResponseTarget(use_entra_auth=True)

attack = PromptSendingAttack(objective_target=target)

result = await attack.execute_async(objective="Tell me a joke")  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %%NBQA-CELL-SEP52c935
from pyrit.models import Message, MessagePiece
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)


async def get_current_weather(args):
    return {
        "weather": "Sunny",
        "temp_c": 22,
        "location": args["location"],
        "unit": args["unit"],
    }


# Responses API function tool schema (flat, no nested "function" key)
function_tool = {
    "type": "function",
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City and state"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location", "unit"],
        "additionalProperties": False,
    },
    "strict": True,
}

# Let the model auto-select tools
target = OpenAIResponseTarget(
    custom_functions={"get_current_weather": get_current_weather},
    extra_body_parameters={
        "tools": [function_tool],
        "tool_choice": "auto",
    },
    httpx_client_kwargs={"timeout": 60.0},
)

# Build the user prompt
message_piece = MessagePiece(
    role="user",
    original_value="What is the weather in Boston in celsius? Use the get_current_weather function.",
    original_value_data_type="text",
)
message = Message(message_pieces=[message_piece])

response = await target.send_prompt_async(message=message)  # type: ignore

for idx, piece in enumerate(response.message_pieces):
    print(f"{idx} | {piece.role}: {piece.original_value}")

# %%NBQA-CELL-SEP52c935
import os

from pyrit.common.tool_configs import web_search_tool
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target.openai.openai_response_target import OpenAIResponseTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

# Note: web search is not yet supported by Azure OpenAI endpoints so we'll use OpenAI from here on.
target = OpenAIResponseTarget(
    endpoint=os.getenv("PLATFORM_OPENAI_RESPONSES_ENDPOINT"),
    api_key=os.getenv("PLATFORM_OPENAI_RESPONSES_KEY"),
    model_name=os.getenv("PLATFORM_OPENAI_RESPONSES_MODEL"),
    extra_body_parameters={
        "tools": [web_search_tool()],
        "tool_choice": "auto",
    },
    httpx_client_kwargs={"timeout": 60},
)

message_piece = MessagePiece(
    role="user", original_value="Briefly, what is one positive news story from today?", original_value_data_type="text"
)
message = Message(message_pieces=[message_piece])

response = await target.send_prompt_async(message=message)  # type: ignore

for idx, piece in enumerate(response.message_pieces):
    if piece.original_value_data_type != "reasoning":
        print(f"{idx} | {piece.role}: {piece.original_value}")

# %%NBQA-CELL-SEP52c935
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)


message_piece = MessagePiece(
    role="user",
    original_value="What is the capital of Italy?",
    original_value_data_type="text",
)
message = Message(message_pieces=[message_piece])

# Define a grammar that prevents "Rome" from being generated
lark_grammar = r"""
start: "I think that it is " SHORTTEXT 
SHORTTEXT: /[^RrOoMmEe]{1,8}/
"""

grammar_tool = {
    "type": "custom",
    "name": "CitiesGrammar",
    "description": "Constrains generation.",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": lark_grammar,
    },
}

target = OpenAIResponseTarget(
    endpoint=os.getenv("PLATFORM_OPENAI_RESPONSES_ENDPOINT"),
    api_key=os.getenv("PLATFORM_OPENAI_RESPONSES_KEY"),
    model_name="gpt-5",
    extra_body_parameters={"tools": [grammar_tool], "tool_choice": "required"},
    temperature=1.0,
)

unconstrained_target = OpenAIResponseTarget(
    endpoint=os.getenv("PLATFORM_OPENAI_RESPONSES_ENDPOINT"),
    api_key=os.getenv("PLATFORM_OPENAI_RESPONSES_KEY"),
    model_name="gpt-5",
    temperature=1.0,
)

unconstrained_result = await unconstrained_target.send_prompt_async(message=message)  # type: ignore

result = await target.send_prompt_async(message=message)  # type: ignore

print("Unconstrained Response:")
for idx, piece in enumerate(unconstrained_result.message_pieces):
    if piece.original_value_data_type != "reasoning":
        print(f"{idx} | {piece.role}: {piece.original_value}")

print()

print("Constrained Response:")
for idx, piece in enumerate(result.message_pieces):
    if piece.original_value_data_type != "reasoning":
        print(f"{idx} | {piece.role}: {piece.original_value}")

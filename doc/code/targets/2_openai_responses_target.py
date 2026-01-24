# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
# ---

# %% [markdown]
# # 2. OpenAI Responses Target
#
# In this demo, we show an example of the `OpenAIResponseTarget`. [Responses](https://platform.openai.com/docs/api-reference/responses) is a newer protocol than chat completions and provides additional functionality with a somewhat modified API. The allowed input types include text, image, web search, file search, functions, reasoning, and computer use.
#
# Before you begin, ensure you are set up with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/populating_secrets.md).
#
#
# ## OpenAI Configuration
#
# Like most targets, all `OpenAITarget`s need an `endpoint` and often also needs a `model` and a `key`. These can be passed into the constructor or configured with environment variables (or in .env).
#
# - endpoint: The API endpoint (`OPENAI_RESPONSES_ENDPOINT` environment variable). For OpenAI, these are just "https://api.openai.com/v1/responses".
# - auth: The API key for authentication (`OPENAI_RESPONSES_KEY` environment variable).
# - model_name: The model to use (`OPENAI_RESPONSES_MODEL` environment variable). For OpenAI, these are any available model name and are listed here: "https://platform.openai.com/docs/models".

# %%
from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

target = OpenAIResponseTarget()
# For Azure OpenAI with Entra ID authentication enabled, use the following command instead. Make sure to run `az login` first.
# from pyrit.auth import get_azure_openai_auth
# endpoint = "https://your-endpoint.openai.azure.com"
# target = OpenAIResponseTarget(
#     endpoint=endpoint,
#     api_key=get_azure_openai_auth(endpoint),
#     model_name="your-deployment-name"
# )

attack = PromptSendingAttack(objective_target=target)

result = await attack.execute_async(objective="Tell me a joke")  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

# %% [markdown]
# ## JSON Generation
#
# We can use the OpenAI `Responses API` with a JSON schema to produce structured JSON output. In this example, we define a simple JSON schema that describes a person with `name` and `age` properties.
#
# For more information about structured outputs with OpenAI, see [the OpenAI documentation](https://platform.openai.com/docs/guides/structured-outputs).

# %%
import json

import jsonschema

from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

# Define a simple JSON schema for a person
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

prompt = "Create a JSON object describing a person named Alice who is 30 years old."
# Create the message piece and message
message_piece = MessagePiece(
    role="user",
    original_value=prompt,
    original_value_data_type="text",
    prompt_metadata={
        "response_format": "json",
        "json_schema": json.dumps(person_schema),
    },
)
message = Message(message_pieces=[message_piece])

# Create the OpenAI Responses target
target = OpenAIResponseTarget()

# Send the prompt, requesting JSON output
response = await target.send_prompt_async(message=message)  # type: ignore

# Validate and print the response
response_json = json.loads(response[0].message_pieces[1].converted_value)
print(json.dumps(response_json, indent=2))
jsonschema.validate(instance=response_json, schema=person_schema)

# %% [markdown]
# ## Tool Use with Custom Functions
#
# In this example, we demonstrate how the OpenAI `Responses API` can be used to invoke a **custom-defined Python function** during a conversation. This is part of OpenAI’s support for "function calling", where the model decides to call a registered function, and the application executes it and passes the result back into the conversation loop.
#
# We define a simple tool called `get_current_weather`, which simulates weather information retrieval. A corresponding OpenAI tool schema describes the function name, parameters, and expected input format.
#
# The function is registered in the `custom_functions` argument of `OpenAIResponseTarget`. The `extra_body_parameters` include:
#
# - `tools`: the full OpenAI tool schema for `get_current_weather`.
# - `tool_choice: "auto"`: instructs the model to decide when to call the function.
#
# The user prompt explicitly asks the model to use the `get_current_weather` function. Once the model responds with a `function_call`, PyRIT executes the function, wraps the output, and the conversation continues until a final answer is produced.
#
# This showcases how agentic function execution works with PyRIT + OpenAI Responses API.

# %%
from pyrit.models import Message, MessagePiece
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore


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

for response_msg in response:
    for idx, piece in enumerate(response_msg.message_pieces):
        print(f"{idx} | {piece.role}: {piece.original_value}")

# %% [markdown]
# ## Using the Built-in Web Search Tool
#
# In this example, we use a built-in PyRIT helper function `web_search_tool()` to register a web search tool with OpenAI's Responses API. This allows the model to issue web search queries during a conversation to supplement its responses with fresh information.
#
# The tool is added to the `extra_body_parameters` passed into the `OpenAIResponseTarget`. As before, `tool_choice="auto"` enables the model to decide when to invoke the tool.
#
# The user prompt asks for a recent positive news story — an open-ended question that may prompt the model to issue a web search tool call. PyRIT will automatically execute the tool and return the output to the model as part of the response.
#
# This example demonstrates how retrieval-augmented generation (RAG) can be enabled in PyRIT through OpenAI's Responses API and integrated tool schema.
#
# NOTE that web search is NOT supported through an Azure OpenAI endpoint, only through the OpenAI platform endpoint (i.e. api.openai.com)

# %%
import os

from pyrit.common.tool_configs import web_search_tool
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

# Note: web search is only supported on a limited set of models.
target = OpenAIResponseTarget(
    endpoint=os.getenv("AZURE_OPENAI_GPT41_RESPONSES_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_GPT41_RESPONSES_KEY"),
    model_name=os.getenv("AZURE_OPENAI_GPT41_RESPONSES_MODEL"),
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

for response_msg in response:
    for idx, piece in enumerate(response_msg.message_pieces):
        print(f"{idx} | {piece.role}: {piece.original_value}")

# %% [markdown]
# ## Grammar-Constrained Generation
#
# OpenAI models also support constrained generation in the [Responses API](https://platform.openai.com/docs/guides/function-calling#context-free-grammars). This forces the LLM to produce output which conforms to the given grammar, which is useful when specific syntax is required in the output.
#
# In this example, we will define a simple Lark grammar which prevents the model from giving a correct answer to a simple question, and compare that to the unconstrained model.
#
# Note that as of October 2025, this is only supported by OpenAI (not Azure) on "gpt-5"

# %%
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

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
    endpoint=os.getenv("AZURE_OPENAI_GPT5_RESPONSES_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_GPT5_KEY"),
    model_name=os.getenv("AZURE_OPENAI_GPT5_MODEL"),
    extra_body_parameters={"tools": [grammar_tool], "tool_choice": "required"},
    temperature=1.0,
)

unconstrained_target = OpenAIResponseTarget(
    endpoint=os.getenv("AZURE_OPENAI_GPT5_RESPONSES_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_GPT5_KEY"),
    model_name=os.getenv("AZURE_OPENAI_GPT5_MODEL"),
    temperature=1.0,
)

unconstrained_result = await unconstrained_target.send_prompt_async(message=message)  # type: ignore

result = await target.send_prompt_async(message=message)  # type: ignore

print("Unconstrained Response:")
for response_msg in unconstrained_result:
    for idx, piece in enumerate(response_msg.message_pieces):
        print(f"{idx} | {piece.role}: {piece.original_value}")

print()

print("Constrained Response:")
for response_msg in result:
    for idx, piece in enumerate(response_msg.message_pieces):
        print(f"{idx} | {piece.role}: {piece.original_value}")

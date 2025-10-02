# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %% [markdown]
# # 8. OpenAI Responses Target
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
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import OpenAIResponseTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

target = OpenAIResponseTarget()

attack = PromptSendingAttack(objective_target=target)

result = await attack.execute_async(objective="Tell me a joke")  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

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
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target.openai.openai_response_target import OpenAIResponseTarget

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
    model_name="o4-mini",
    custom_functions={"get_current_weather": get_current_weather},
    extra_body_parameters={
        "tools": [function_tool],
        "tool_choice": "auto",
    },
    httpx_client_kwargs={"timeout": 60.0},
    # api_version=None,  # this can be uncommented if using api.openai.com
)

# Build the user prompt
prompt_piece = PromptRequestPiece(
    role="user",
    original_value="What is the weather in Boston in celsius? Use the get_current_weather function.",
    original_value_data_type="text",
)
prompt_request = PromptRequestResponse(request_pieces=[prompt_piece])

response = await target.send_prompt_async(prompt_request=prompt_request)  # type: ignore

for idx, piece in enumerate(response.request_pieces):
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
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.common.tool_configs import web_search_tool
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target.openai.openai_response_target import OpenAIResponseTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

target = OpenAIResponseTarget(
    extra_body_parameters={
        "tools": [web_search_tool()],
        "tool_choice": "auto",
    },
    httpx_client_kwargs={"timeout": 60},
)

prompt_piece = PromptRequestPiece(
    role="user", original_value="Briefly, what is one positive news story from today?", original_value_data_type="text"
)
prompt_request = PromptRequestResponse(request_pieces=[prompt_piece])

response = await target.send_prompt_async(prompt_request=prompt_request)  # type: ignore

for idx, piece in enumerate(response.request_pieces):
    # Reasoning traces are necessary to be sent back to the endpoint for function calling even if they're empty.
    # They are excluded here for a cleaner output.
    if piece.original_value_data_type != "reasoning":
        print(f"{idx} | {piece.role}: {piece.original_value}")

# %% [markdown]
# ## Azure OpenAI GPT4-o Target Demo
# This notebook demonstrates how to use the Azure OpenAI GPT4-o target to accept multimodal input (text+image) and generate text output.

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import AzureOpenAIGPT4OChatTarget
from pyrit.common import default_values
import pathlib
from pyrit.common.path import HOME_PATH
import uuid

default_values.load_default_env()
test_conversation_id = str(uuid.uuid4())

# use the image from our docs
image_path = pathlib.Path(HOME_PATH) / "assets" / "pyrit_architecture.png"

request_pieces = [
    PromptRequestPiece(
        role="user",
        conversation_id=test_conversation_id,
        original_value="Describe this picture:",
        original_value_data_type="text",
        converted_value_data_type="text",
    ),
    PromptRequestPiece(
        role="user",
        conversation_id=test_conversation_id,
        original_value=str(image_path),
        original_value_data_type="image_path",
        converted_value_data_type="image_path",
    ),
]

# %%
prompt_request_response = PromptRequestResponse(request_pieces=request_pieces)

# %%
with AzureOpenAIGPT4OChatTarget() as azure_openai_chat_target:
    resp = await azure_openai_chat_target.send_prompt_async(prompt_request=prompt_request_response)  # type: ignore
    print(resp)

# %%

# %% [markdown]
# ## Multimodal Demo using AzureOpenAIGPT4OChatTarget and PromptSendingOrchestrator
# This demo showcases the capabilities of AzureOpenAIGPT4OChatTarget for generating text based on multimodal inputs, including both text and images using PromptSendingOrchestrator.

# %%
from pyrit.common import default_values
import pathlib
from pyrit.common.path import HOME_PATH

from pyrit.prompt_target import AzureOpenAIGPT4OChatTarget
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequestPiece
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.orchestrator import PromptSendingOrchestrator

default_values.load_default_env()

azure_openai_gpt4o_chat_target = AzureOpenAIGPT4OChatTarget()

image_path = pathlib.Path(HOME_PATH) / "assets" / "pyrit_architecture.png"
data = [
    [
        {"prompt_text": "Describe this picture:", "prompt_data_type": "text"},
        {"prompt_text": str(image_path), "prompt_data_type": "image_path"},
    ],
    [{"prompt_text": "Tell me about something?", "prompt_data_type": "text"}],
    [{"prompt_text": str(image_path), "prompt_data_type": "image_path"}],
]

# %% [markdown]
# Construct list of NormalizerRequest objects

# %%

normalizer_requests = []

for piece_data in data:
    request_pieces = []

    for item in piece_data:
        prompt_text = item.get("prompt_text", "")  # type: ignore
        prompt_data_type = item.get("prompt_data_type", "")
        converters = []  # type: ignore
        request_piece = NormalizerRequestPiece(
            prompt_value=prompt_text, prompt_data_type=prompt_data_type, request_converters=converters  # type: ignore
        )
        request_pieces.append(request_piece)  # type: ignore

    normalizer_request = NormalizerRequest(request_pieces)  # type: ignore
    normalizer_requests.append(normalizer_request)

# %%
len(normalizer_requests)

# %%

with PromptSendingOrchestrator(prompt_target=azure_openai_gpt4o_chat_target) as orchestrator:

    await orchestrator.send_normalizer_requests_async(prompt_request_list=normalizer_requests)  # type: ignore

    memory = orchestrator.get_memory()

    for entry in memory:
        print(entry)

# %%

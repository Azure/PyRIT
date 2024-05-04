# %% [markdown]
# ## Azure OpenAI GPT-V Target Demo
# This notebook demonstrates how to use the Azure OpenAI GPT-V target to accept multimodal input (text+image) and generate text output.

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import AzureOpenAIGPTVChatTarget
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
with AzureOpenAIGPTVChatTarget() as azure_openai_chat_target:
    resp = await azure_openai_chat_target.send_prompt_async(prompt_request=prompt_request_response)  # type: ignore
    print(resp)

# %%

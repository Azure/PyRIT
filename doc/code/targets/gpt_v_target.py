# %% [markdown]
# ## Azure OpenAI GPT-V Target Demo
# This notebook demonstrates how to use the Azure OpenAI GPT-V target to accept multimodal input (text+image) and generate text output.

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import AzureOpenAIGPTVChatTarget
from pyrit.common import default_values
import uuid

default_values.load_default_env()
test_conversation_id = str(uuid.uuid4())

request_pieces = [
PromptRequestPiece(
    role="user",
    conversation_id=test_conversation_id,
    original_prompt_text="Describe this picture:",
    original_prompt_data_type="text",
    converted_prompt_data_type="text"
), 
PromptRequestPiece(
    role="user",
    conversation_id=test_conversation_id,
    original_prompt_text="C://data//images//aeroplane.jpg",
    original_prompt_data_type="image_path",
    converted_prompt_data_type="image_path"
)]

# %%
prompt_request_response = PromptRequestResponse(request_pieces=request_pieces)

# %%
with AzureOpenAIGPTVChatTarget() as azure_openai_chat_target:
    resp = await azure_openai_chat_target.send_prompt_async(prompt_request=prompt_request_response)
    print(resp)

# %%




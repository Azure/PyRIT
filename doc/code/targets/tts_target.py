# %% [markdown]
# ## Image Target Demo
# This notebook demonstrates how to use the TTS (audio) target to convert text to speech

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptRequestPiece
from pyrit.prompt_target import AzureTTSTarget
from pyrit.common import default_values

default_values.load_default_env()

request = PromptRequestPiece(
    role="user",
    original_prompt_text="Hello, I am an audio prompt",
).to_prompt_request_response()


with AzureTTSTarget() as azure_openai_chat_target:
    resp = await azure_openai_chat_target.send_prompt_async(prompt_request=request)  # type: ignore

    # The response is an mp3 saved to disk (but also included as part of memory)
    print(resp)

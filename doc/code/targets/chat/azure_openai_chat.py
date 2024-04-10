# %% [markdown]
#
# # Introduction
#
# This Jupyter notebook gives an introduction on how to use `AzureOpenAIChat` to complete chats.
#
# Before starting this, make sure you are [setup to use Azure OpenAI endpoints](../../../setup/setup_azure.md) and have a chat model, such as GPT-4, deployed. See [How To: Create and deploy an Azure OpenAI Service resource](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).
#
# In this case, we have one named `gpt-4` deployed. See your deployments at https://oai.azure.com/ `> Management > Deployments`

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid

from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.common import default_values

default_values.load_default_env()


with AzureOpenAIChatTarget() as azure_openai_chat_target:
    azure_openai_chat_target.send_prompt(
        normalized_prompt="Hello world!", conversation_id=str(uuid.uuid4()), normalizer_id=None
    )

# %%

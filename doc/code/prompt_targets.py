# %% [markdown]
# Prompt Targets are endpoints for where to send prompts.
# In this demo, we show examples of the `AzureOpenAIChatTarget` and the `AzureBlobStorageTarget`.
#
# The `AzureBlobStorageTarget` inherits from `PromptTarget`, meaning it has functionality to send prompts.
# This prompt target in particular will take in a prompt, convert it to a text file, and upload it as a blob to the provided Azure Storage Account Container.
# This could be useful for XPIA scenarios, for example, where there is a jailbreak within a file.
#
# Note: to run this section of the demo you need to setup your `.env` file to properly authenticate to an Azure Storage Blob Container.
# See the section within [.env_example](https://github.com/Azure/PyRIT/blob/main/.env_example) if not sure where to find values for each of these variables.
# %%

import os

from pyrit.prompt_target import AzureBlobStorageTarget
from pyrit.common import default_values


default_values.load_default_env()

abs_prompt_target = AzureBlobStorageTarget(
    container_url=os.environ.get("AZURE_STORAGE_ACCOUNT_CONTAINER_URL"),
    sas_token=os.environ.get("AZURE_STORAGE_ACCOUNT_SAS_TOKEN"),
)


# %% [markdown]
# The `AzureOpenAIChatTarget` inherits from the `PromptChatTarget` class, which expands upon the `PromptTarget` class by adding functionality to set a system prompt.
#
#  Note: to run this section of the demo you need to setup your `.env` file to properly authenticate to an Azure OpenAI endpoint as described [here](../setup/setup_azure.md)
# %%

import os

from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.common import default_values


default_values.load_default_env()

aoi_prompt_target = AzureOpenAIChatTarget(
    deployment_name="defense-gpt35",
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
)

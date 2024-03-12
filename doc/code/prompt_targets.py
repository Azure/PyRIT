# %% [markdown]
# ## Prompt Targets
# Prompt Targets are endpoints for where to send prompts.
# In this demo, we show examples of the `AzureOpenAIChatTarget` and the `AzureBlobStorageTarget`.
#
# Prompt Targets are typically used with [orchestrators](https://github.com/Azure/PyRIT/blob/main/doc/code/orchestrator.ipynb), but will be shown individually here.
#
# The `AzureBlobStorageTarget` inherits from `PromptTarget`, meaning it has functionality to send prompts.
# This prompt target in particular will take in a prompt and upload it as a text file to the provided Azure Storage Account Container.
# This could be useful for XPIA scenarios, for example, where there is a jailbreak within a file.
#
# Note: to run this section of the demo you need to setup your `.env` file to properly authenticate to an Azure Storage Blob Container.
# See the section within [.env_example](https://github.com/Azure/PyRIT/blob/main/.env_example) if not sure where to find values for each of these variables.
# %%

import os
import uuid

from pyrit.prompt_target import AzureBlobStorageTarget
from pyrit.common import default_values


# When using a Prompt Target with an Orchestrator, conversation ID and normalizer ID are handled for you
test_conversation_id = str(uuid.uuid4())
test_normalizer_id = "1"

default_values.load_default_env()

abs_prompt_target = AzureBlobStorageTarget(
    container_url=os.environ.get("AZURE_STORAGE_ACCOUNT_CONTAINER_URL"),
    sas_token=os.environ.get("AZURE_STORAGE_ACCOUNT_SAS_TOKEN"),
)

abs_prompt_target.send_prompt(
    normalized_prompt="This contains a cool jailbreak that has been converted as specified with prompt converters!",
    conversation_id=test_conversation_id,
    normalizer_id=test_normalizer_id,
)

# TODO: Example with list_all_blob_urls() if it is useful.


# Alternatively, send prompts asynchronously
async def send_prompt_async_example_abs():
    await abs_prompt_target.send_prompt_async(
        normalized_prompt="This contains a cool jailbreak that has been converted as specified with prompt converters!",
        conversation_id=test_conversation_id,
        normalizer_id=test_normalizer_id,
    )


# %% [markdown]
# The `AzureOpenAIChatTarget` inherits from the `PromptChatTarget` class, which expands upon the `PromptTarget` class by adding functionality to set a system prompt.
#
#  Note: to run this section of the demo you need to setup your `.env` file to properly authenticate to an Azure OpenAI endpoint as described [here](../setup/setup_azure.md)
# %%

import os

from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.common import default_values


# When using a Prompt Target with an Orchestrator, conversation ID and normalizer ID are handled for you
test_conversation_id = str(uuid.uuid4())
test_normalizer_id = "1"

default_values.load_default_env()

aoi_prompt_target = AzureOpenAIChatTarget(
    deployment_name="defense-gpt35",
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
)

aoi_prompt_target.set_system_prompt(
    prompt="System Instructions to set as Prompt",
    conversation_id=test_conversation_id,
    normalizer_id=test_normalizer_id,
)

aoi_prompt_target.send_prompt(
    normalized_prompt="This contains a cool jailbreak that has been converted as specified with prompt converters!",
    conversation_id=test_conversation_id,
    normalizer_id=test_normalizer_id,
)


# Alternatively, send prompts asynchronously
async def send_prompt_async_example_aoi():
    await aoi_prompt_target.send_prompt_async(
        normalized_prompt="This contains a cool jailbreak that has been converted as specified with prompt converters!",
        conversation_id=test_conversation_id,
        normalizer_id=test_normalizer_id,
    )

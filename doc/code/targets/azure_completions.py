# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Azure OpenAI Completions
#
# Before you begin, ensure you are setup with the correct version of PyRIT and have the applicable secrets configured as described [here](../../setup/).
#
# Once you are configured, then you will be able to get completions for your text.

# %%
import os
from pyrit.common import default_values

from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import AzureOpenAICompletionTarget

default_values.load_default_env()

api_base = os.environ.get("AZURE_OPENAI_COMPLETION_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_COMPLETION_KEY")
deployment_name = os.environ.get("AZURE_OPENAI_COMPLETION_DEPLOYMENT")


target = AzureOpenAICompletionTarget(api_key=api_key, endpoint=api_base, deployment_name=deployment_name)

with PromptSendingOrchestrator(prompt_target=target) as orchestrator:
    response = await orchestrator.send_prompts_async(prompt_list=["Hello! Who are you?"])  # type: ignore
    print(response[0])

# %%

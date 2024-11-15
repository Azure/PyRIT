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
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # OpenAI Completions - optional
#
# Before you begin, ensure you are setup with the correct version of PyRIT and have the applicable secrets configured as described [here](../../setup/populating_secrets.md).
#
# Once you are configured, then you will be able to get completions for your text.

# %%
from pyrit.common import default_values

from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAICompletionTarget

default_values.load_environment_files()

# Note that max_tokens will default to 16 for completions, so you may want to set the upper limit of allowed tokens for a longer response.
target = OpenAICompletionTarget(max_tokens=2048)

with PromptSendingOrchestrator(prompt_target=target) as orchestrator:
    response = await orchestrator.send_prompts_async(prompt_list=["Hello! Who are you?"])  # type: ignore
    print(response[0])

# %%

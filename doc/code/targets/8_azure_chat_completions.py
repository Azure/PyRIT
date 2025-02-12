# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: pyrit-312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Azure Chat Completions Target
#
# This will allow you to easily interact with models in [Azure AI Foundry](https://ai.azure.com/), such as DeepSeek R1, Phi-4, Mistral, and LLAMA.
#
# Before you begin, ensure you are setup with the correct version of PyRIT installed and have secrets configured as described [here](../../setup/populating_secrets.md).

# %%
from pyrit.datasets.fetch_example_datasets import fetch_harmbench_dataset

seed_prompts = fetch_harmbench_dataset(cache=True)
text_prompts = [v.value for v in seed_prompts.prompts[:3]]

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target.azure_chat_completions_target import AzureChatCompletionsTarget

initialize_pyrit(memory_db_type=IN_MEMORY)


phi4_target = AzureChatCompletionsTarget()
orchestrator = PromptSendingOrchestrator(objective_target=phi4_target)

response = await orchestrator.send_prompts_async(prompt_list=text_prompts)  # type: ignore
await orchestrator.print_conversations_async()  # type: ignore

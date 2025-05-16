# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fetching dataset examples
#
# This notebook demonstrates how to load datasets as a `SeedPromptDataset` to perform red teaming on a target.
# There are several datasets which can be found in the `pyrit.datasets` module.
# Three example datasets are shown in this notebook and can be used with orchestrators such as the Prompt Sending Orchestrator.
# The example below demonstrates loading a HuggingFace dataset as a `SeedPromptDataset`.

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.datasets import fetch_llm_latent_adversarial_training_harmful_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()

# Fetch prompt column from harmful datasets
prompt_dataset = fetch_llm_latent_adversarial_training_harmful_dataset()
# Some fetch functions may include parameters such as the below example for unsafe prompts
# prompt_dataset = fetch_pku_safe_rlhf_dataset(False)

# Use the first 8 examples for red teaming
prompt_list = prompt_dataset.get_values(first=8)

# Send prompts using the orchestrator and capture responses
orchestrator = PromptSendingOrchestrator(objective_target=prompt_target)
responses = await orchestrator.run_attacks_async(objectives=prompt_list)  # type: ignore

# %% [markdown]
# # Example dataset from public URL
#
# The following example fetches DecodingTrust 'stereotypes' examples of involving potentially harmful stereotypes from the DecodingTrust repository which try to convince the assistant to agree and captures the responses. This is a scenario where the dataset resides in a public  URL and is also outputted as a `SeedPromptDataset`. By fetching these prompts, we can further use this `SeedPromptDataset` by sending the prompts to a target using the `PromptSendingOrchestrator` as shown in the example below.

# %%
from pyrit.datasets import fetch_decoding_trust_stereotypes_dataset
from pyrit.prompt_target import TextTarget

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()
examples_source = (
    "https://raw.githubusercontent.com/AI-secure/DecodingTrust/main/data/stereotype/dataset/user_prompts.csv"
)

orchestrator = PromptSendingOrchestrator(objective_target=prompt_target)

# Fetch examples from DecodingTrust 'Stereotype' dataset using the 'targeted' system prompt and topics of "driving" and "technology"
prompt_dataset = fetch_decoding_trust_stereotypes_dataset(
    examples_source,
    source_type="public_url",
    stereotype_topics=["driving", "technology"],
    target_groups=None,
    system_prompt_type="targeted",
)

# Use the first 4 examples
prompt_list = prompt_dataset.get_values(first=4)

responses = await orchestrator.run_attacks_async(objectives=prompt_list)  # type: ignore

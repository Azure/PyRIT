# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TDC-23 Red Teaming Dataset - optional
#
# This notebook demonstrates the process of using examples from the TDC-23 Red Teaming dataset to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts from the HugginFace source, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of harmful or ethically sensitive prompts.

# %%
from pyrit.common.initialize_pyrit import initialize_pyrit, IN_MEMORY
from pyrit.datasets import fetch_tdc23_redteaming_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget


initialize_pyrit(memory_db_type=IN_MEMORY)

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()

# Note:
# The dataset sources can be found at:
# - HuggingFace source: https://huggingface.co/datasets/walledai/TDC23-RedTeaming

# Fetch only unsafe prompts from tdc23_redteaming dataset
prompt_dataset = fetch_tdc23_redteaming_dataset()

# Use the first 8 examples for red teaming
prompt_list = [prompt.value for prompt in prompt_dataset.prompts[:8]]

# Send prompts using the orchestrator and capture responses
orchestrator = PromptSendingOrchestrator(objective_target=prompt_target)
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

# Close connection
orchestrator.dispose_db_engine()

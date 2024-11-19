# ---
# jupyter:
#   jupytext:
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
# # PKU-SafeRLHF dataset testing - optional
#
# This notebook demonstrates the process of using PKU-SafeRLHF examples to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts from the PKU-SafeRLHF repository, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of harmful or ethically sensitive prompts.
#

# %% [markdown]
#

# %%
# Import necessary packages
from pyrit.common import default_values
from pyrit.datasets import fetch_pku_safe_rlhf_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

# %%
# Load environment variables
default_values.load_environment_files()

# %%
# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()

# %%
# Create the orchestrator with scorer without safe prompts included
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

# Fetch only unsafe prompts from PKU-SafeRLHF dataset
prompt_dataset = fetch_pku_safe_rlhf_dataset(False)

# Use the first 8 examples for red teaming
prompt_list = prompt_dataset.prompts[:8]

# Send prompts using the orchestrator and capture responses
try:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
    if responses:
        orchestrator.print_conversations()  # Retrieve the memory to print scoring results
    else:
        print("No valid responses were received from the orchestrator.")
except Exception as e:
    print(f"An error occurred while sending prompts: {e}")


# %%
# Create the orchestrator with scorer with safe prompts included
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

# Fetch prompts from PKU-SafeRLHF dataset
prompt_dataset = fetch_pku_safe_rlhf_dataset(True)

# Use the first 8 examples for red teaming
prompt_list = prompt_dataset.prompts[:8]

# Send prompts using the orchestrator and capture responses
try:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
    if responses:
        orchestrator.print_conversations()  # Retrieve the memory to print scoring results
    else:
        print("No valid responses were received from the orchestrator.")
except Exception as e:
    print(f"An error occurred while sending prompts: {e}")

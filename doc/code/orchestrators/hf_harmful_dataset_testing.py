# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM-LAT/harmful-dataset testing - optional
#
# This notebook demonstrates the testing of import of huggingface dataset "https://huggingface.co/datasets/LLM-LAT/harmful-dataset"

# %% [markdown]
#

# %%
# Import necessary packages
from pyrit.common import default_values
from pyrit.datasets import fetch_llm_latent_adversarial_training_harmful_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

# %%
# Load environment variables
default_values.load_environment_files()

# %%
# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()
example_source = "https://huggingface.co/datasets/LLM-LAT/harmful-dataset"

# %%
# Create the orchestrator with scorer
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

# Fetch prompt column from harmful-datasets
prompt_dataset = fetch_llm_latent_adversarial_training_harmful_dataset()

# Use the first 8 examples for red teaming
prompt_list = prompt_dataset.prompts[:8]

# Send prompts using the orchestrator and capture responses
try:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
    if responses:
        await orchestrator.print_conversations()  # type: ignore
    else:
        print("No valid responses were received from the orchestrator.")
except Exception as e:
    print(f"An error occurred while sending prompts: {e}")

# %%

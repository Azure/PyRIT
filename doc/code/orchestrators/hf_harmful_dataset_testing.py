# ---
# jupyter:
#   jupytext:
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
# # LLM-LAT/harmful-dataset testing - optional
#
# This notebook demonstrates the testing of import of huggingface dataset "https://huggingface.co/datasets/LLM-LAT/harmful-dataset"

# %%
from pyrit.common import default_values
from pyrit.datasets import fetch_llm_latent_adversarial_training_harmful_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

# Load environment variables
default_values.load_environment_files()

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()
example_source = "https://huggingface.co/datasets/LLM-LAT/harmful-dataset"

# Fetch prompt column from harmful-datasets
prompt_dataset = fetch_llm_latent_adversarial_training_harmful_dataset()

# Use the first 8 examples for red teaming
prompt_list = [prompt.value for prompt in prompt_dataset.prompts[:8]]

# Send prompts using the orchestrator and capture responses
with PromptSendingOrchestrator(objective_target=prompt_target) as orchestrator:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

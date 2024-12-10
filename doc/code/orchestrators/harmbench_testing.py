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
# # HarmBench dataset testing - optional
#
# This notebook demonstrates the process of using HarmBench examples to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts from the HarmBench repository, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of harmful or ethically sensitive prompts.


# %% [markdown]
# Note:
# The dataset sources can be found at:
# - GitHub repository: https://github.com/centerforaisafety/HarmBench


# %%
from pyrit.common import default_values
from pyrit.datasets import fetch_harmbench_examples
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

default_values.load_environment_files()

prompt_target = TextTarget()
examples_source = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/harmbench_behaviors_text_all.csv"

# Fetch examples from HarmBench dataset
prompt_dataset = fetch_harmbench_examples(examples_source, source_type="public_url")

# Use the first 4 examples for red teaming
prompt_list = [prompt.value for prompt in prompt_dataset.prompts[:4]]

with PromptSendingOrchestrator(objective_target=prompt_target) as orchestrator:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

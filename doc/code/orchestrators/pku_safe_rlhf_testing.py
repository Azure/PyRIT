# ---
# jupyter:
#   jupytext:
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
# # PKU-SafeRLHF dataset testing - optional
#
# This notebook demonstrates the process of using PKU-SafeRLHF examples to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts from the PKU-SafeRLHF repository, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of harmful or ethically sensitive prompts.
#

# %%
from pyrit.common import initialize_pyrit, IN_MEMORY
from pyrit.datasets import fetch_pku_safe_rlhf_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget


initialize_pyrit(memory_db_type=IN_MEMORY)

prompt_target = TextTarget()

# Fetch only unsafe prompts from PKU-SafeRLHF dataset
prompt_dataset = fetch_pku_safe_rlhf_dataset(False)

# Use the first 8 examples for red teaming
prompt_list = [prompt.value for prompt in prompt_dataset.prompts[:8]]

orchestrator = PromptSendingOrchestrator(objective_target=prompt_target)
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

# %%
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()

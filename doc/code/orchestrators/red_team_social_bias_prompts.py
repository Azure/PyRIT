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
# # Red Team Social Bias dataset testing
#
# This dataset aggregates and unifies existing red-teaming prompts
# designed to identify stereotypes, discrimination, hate speech, and
# other representation harms in text-based Large Language Models (LLMs).

# %%
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.datasets import fetch_red_team_social_bias_prompts_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

initialize_pyrit(memory_db_type=IN_MEMORY)

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()

# Fetch prompt column from harmful datasets
prompt_dataset = fetch_red_team_social_bias_prompts_dataset()
# Some fetch functions may include parameters such as the below example for unsafe prompts
# prompt_dataset = fetch_pku_safe_rlhf_dataset(False)

# Use the first 8 examples for red teaming
prompt_list = [prompt.value for prompt in prompt_dataset.prompts[:8]]

# Send prompts using the orchestrator and capture responses
orchestrator = PromptSendingOrchestrator(objective_target=prompt_target)
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

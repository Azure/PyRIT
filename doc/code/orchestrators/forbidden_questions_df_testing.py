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
# # TrustAIRLab Forbidden Questions Dataset - optional
#
# This notebook demonstrates the testing of imports of the TrustAIRLab Forbidden Questions Dataset.
# It involves fetching potentially harmful behavior prompts from the HuggingFace source, sending them to the LLM, and analyzing the responses.
# The goal is to identify questions that come under the following harmful categories: Illegal Activity, Hate Speech, Malware Generation, Physical Harm, Economic Harm, Fraud,Pornography, Political Lobbying, Privacy Violence, Legal Opinion, Financial Advice,Health Consultation and Government Decision.

# %%
# Import necessary packages
from pyrit.common import initialize_pyrit, IN_MEMORY
from pyrit.datasets import fetch_forbidden_questions_df
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget


initialize_pyrit(memory_db_type=IN_MEMORY)

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()
example_source = "https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set"


# Fetch prompt column from harmful-datasets
prompt_dataset = fetch_forbidden_questions_df()

# Use the first 8 examples for red teaming
prompt_list = [prompt.value for prompt in prompt_dataset.prompts[:8]]


orchestrator = PromptSendingOrchestrator(objective_target=prompt_target)
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

# Close connection
orchestrator.dispose_db_engine()

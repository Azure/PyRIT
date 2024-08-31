# %% [markdown]
# # HarmBench dataset testing
#
# This notebook demonstrates the process of using HarmBench examples to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts from the HarmBench repository, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of harmful or ethically sensitive prompts.

# %%
# Import necessary packages
import os

from pyrit.common import default_values
from pyrit.datasets import fetch_harmbench_examples
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget
from pyrit.score import SelfAskLikertScorer, LikertScalePaths


# %%
# Load environment variables
default_values.load_default_env()


# %%
# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()
examples_source = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/harmbench_behaviors_text_all.csv"


# %%
# Note:
# The dataset sources can be found at:
# - GitHub repository: https://github.com/centerforaisafety/HarmBench


# %%
# Set up the scorer with the same prompt target to ensure consistency
# The scoring here is to assess how well the model defends against harmful prompts
harm_scorer = SelfAskLikertScorer(likert_scale_path=LikertScalePaths.HARM_SCALE.value, chat_target=prompt_target)

# Create the orchestrator with scorer
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target, scorers=[harm_scorer])

# Fetch examples from HarmBench dataset
prompt_dataset = fetch_harmbench_examples(examples_source, source_type="public_url")

# Send prompts using the orchestrator and capture responses
try:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
except Exception as e:
    print(f"An error occurred while sending prompts: {e}")
    responses = []

# Ensure we handle the responses correctly
if responses:
    # Access memory after sending prompts
    memory = orchestrator.get_memory()
    
    for entry in memory:
        print(entry)
else:
    print("No valid responses were received from the orchestrator.")


# %%

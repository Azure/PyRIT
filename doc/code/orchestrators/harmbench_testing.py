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
from pyrit.prompt_target import AzureOpenAITextChatTarget
from pyrit.score import SelfAskLikertScorer, LikertScalePaths


# %%
# Load environment variables
default_values.load_default_env()


# %%
# Set up the Azure OpenAI prompt target
prompt_target = AzureOpenAITextChatTarget(
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
)
examples_source = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/harmbench_behaviors_text_all.csv"


# %%
# Note:
# The dataset sources can be found at:
# - GitHub repository: https://github.com/centerforaisafety/HarmBench


# %%
# Set up the scorer with the same prompt target to ensure consistency
# The scoring here is to assess how well the model defends against harmful prompts
harm_scorer = SelfAskLikertScorer(
    likert_scale_path=LikertScalePaths.HARM_SCALE.value, chat_target=prompt_target
)

# Create the orchestrator with scorer
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target, scorers=[harm_scorer])

# Fetch examples from HarmBench dataset
prompt_dataset = fetch_harmbench_examples(examples_source, source_type="public_url")

# Use the first 4 examples for red teaming
prompt_list = prompt_dataset.prompts[:4]

# Send prompts using the orchestrator and capture responses
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

# Print the conversations to analyze how the LLM handled the harmful prompts
orchestrator.print_conversations()  # Use built-in method to display conversations

# %%

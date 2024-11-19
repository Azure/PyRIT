# %% [markdown]
# # XSTest Bias Testing - optional
#
# This notebook demonstrates the process of using XSTest Bias Testing examples to evaluate the safety and bias of an AI language model.
# It includes fetching bias testing prompts from the XSTest repository, sending them to the model, and scoring the model's responses based on predefined fairness and bias scales.
# The goal is to assess how well the AI model handles safety without showing exaggerated refusals for safe prompts.


# %%
# Import necessary packages
import os

from pyrit.common import default_values
from pyrit.datasets import fetch_xstest_examples
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskLikertScorer, LikertScalePaths


# %%
# Load environment variables
default_values.load_environment_files()


# %%
prompt_target = OpenAIChatTarget(
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
)
examples_source = "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/a3bb396/xstest_v2_prompts.csv"


# %%
# Note:
# The dataset sources can be found at:
# - GitHub repository: https://github.com/paul-rottger/exaggerated-safety


# %%
# Set up the Bias Scorer with the same prompt target to ensure consistency
bias_scorer = SelfAskLikertScorer(
    likert_scale_path=LikertScalePaths.FAIRNESS_BIAS_SCALE.value, chat_target=prompt_target
)

# Create the orchestrator with scorer
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target, scorers=[bias_scorer])

# Fetch examples from XSTest Bias Testing dataset
prompt_dataset = fetch_xstest_examples(examples_source, source_type="public_url")

# Use the first 4 examples
prompt_list = prompt_dataset.prompts[:4]

# Send prompts using the orchestrator
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

# Retrieve the memory to print scoring results
orchestrator.print_conversations()  # Use built-in method to display conversations

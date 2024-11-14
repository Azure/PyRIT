# %% [markdown]
# # SecLists Bias Testing - optional
#
# This notebook demonstrates the process of using SecLists AI LLM Bias Testing examples to evaluate the bias of an AI language model.
# It includes fetching bias testing prompts from the SecLists repository, sending them to the model, and scoring the model's responses based on predefined fairness and bias scales.
# The goal is to assess how well the AI model handles various biases in its responses.


# %%
# Import necessary packages
from pyrit.common import default_values
from pyrit.datasets import fetch_seclists_bias_testing_examples
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import SelfAskLikertScorer, LikertScalePaths


# %%
# Load environment variables
default_values.load_environment_files()


# %%
prompt_target = OpenAIChatTarget()
examples_source = "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Ai/LLM_Testing/Bias_Testing/nationality_geographic_bias.txt"


# %%
# Note:
# The dataset sources can be found at:
# - GitHub repository: https://github.com/danielmiessler/SecLists/tree/master/Ai/LLM_Testing/Bias_Testing


# %%
# Set up the Bias Scorer with the same prompt target to ensure consistency
bias_scorer = SelfAskLikertScorer(
    likert_scale_path=LikertScalePaths.FAIRNESS_BIAS_SCALE.value, chat_target=prompt_target
)

# Create the orchestrator with scorer
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target, scorers=[bias_scorer])

# Fetch examples from SecLists Bias Testing dataset
prompt_dataset = fetch_seclists_bias_testing_examples(examples_source, source_type="public_url")

# Use the first 5 examples
prompt_list = prompt_dataset.prompts[:4]

# Send prompts using the orchestrator
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

# Retrieve the memory to print scoring results
orchestrator.print_conversations()  # Use built-in method to display conversations


# %%

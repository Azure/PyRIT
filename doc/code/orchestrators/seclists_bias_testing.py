# %% [markdown]
# # SecLists Bias Testing
#
# This notebook demonstrates the process of using SecLists AI LLM Bias Testing examples to evaluate the bias of an AI language model.
# It includes fetching bias testing prompts from the SecLists repository, sending them to the model, and scoring the model's responses based on predefined fairness and bias scales.
# The goal is to assess how well the AI model handles various biases in its responses.


# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# %%
# Import necessary packages
import os
import time
import uuid

from pyrit.common import default_values
from pyrit.datasets.fetch_example_datasets import fetch_seclists_bias_testing_examples
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.score.self_ask_likert_scorer import SelfAskLikertScorer, LikertScalePaths


# %%
# Load environment variables
default_values.load_default_env()


# %%
prompt_target = AzureOpenAIChatTarget(
    api_key=os.environ.get("AZURE_OPENAI_CHAT_KEY"),
    endpoint=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"),
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
)
examples_source = os.getenv("EXAMPLES_SOURCE")


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

# Use only 4 examples
prompt_list = prompt_dataset.prompts[3:7]

# Store original prompts and responses
response_data = {}

# Assign unique IDs to each prompt
prompt_requests = []
for prompt in prompt_list:
    prompt_id = str(uuid.uuid4())
    prompt_requests.append({"id": prompt_id, "prompt": prompt})

# Implement simple retry logic
max_retries = 5
for attempt in range(max_retries):
    try:
        responses = await orchestrator.send_prompts_async(prompt_list=[req["prompt"] for req in prompt_requests]) # type: ignore
        break  # Exit loop if request is successful
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(2**attempt)  # Exponential backoff
        else:
            raise

# Retrieve the memory to print scoring results
orchestrator.print_conversations()  # Use built-in method to display conversations


# %%
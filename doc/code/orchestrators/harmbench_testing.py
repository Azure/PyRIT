# %% [markdown]
# # HarmBench dataset testing - optional
#
# This notebook demonstrates the process of using HarmBench examples to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts from the HarmBench repository, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of harmful or ethically sensitive prompts.


# %%
# Import necessary packages
from pyrit.common import default_values
from pyrit.datasets import fetch_harmbench_examples
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget


# %%
# Load environment variables
default_values.load_environment_files()


# %%
# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()
examples_source = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/c0423b9/data/behavior_datasets/harmbench_behaviors_text_all.csv"


# %% [markdown]
# Note:
# The dataset sources can be found at:
# - GitHub repository: https://github.com/centerforaisafety/HarmBench


# %%
# Create the orchestrator with scorer
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

# Fetch examples from HarmBench dataset
prompt_dataset = fetch_harmbench_examples(examples_source, source_type="public_url")

# Use the first 4 examples for red teaming
prompt_list = prompt_dataset.prompts[:4]

# Send prompts using the orchestrator and capture responses
try:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
    if responses:
        orchestrator.print_conversations()  # Retrieve the memory to print scoring results
    else:
        print("No valid responses were received from the orchestrator.")
except Exception as e:
    print(f"An error occurred while sending prompts: {e}")

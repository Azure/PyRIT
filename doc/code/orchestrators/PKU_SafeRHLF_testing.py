# %% [markdown]
# # PKU-SafeRHLF dataset testing
#
# This notebook demonstrates the process of using PKU-SafeRHLF examples to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts from the PKU-SafeRHLF repository, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of harmful or ethically sensitive prompts.


# %%
# Import necessary packages
from pyrit.common import default_values
from pyrit.datasets import fetch_pku_safeRLHF_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget


# %%
# Load environment variables
default_values.load_default_env()


# %%
# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()

# %%
# Note:
# The dataset sources can be found at:
# - GitHub repository: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF


# %%
# Create the orchestrator with scorer
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

# Fetch examples from HarmBench dataset
prompt_dataset = fetch_pku_safeRLHF_dataset(False)

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


# %%

# %% [markdown]
# # TDC-23 Red Teaming Dataset - optional
#
# This notebook demonstrates the process of using examples from the TDC-23 Red Teaming dataset to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts from the HugginFace source, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of harmful or ethically sensitive prompts.

# %%
from pyrit.common import default_values
from pyrit.datasets import fetch_tdc23_redteaming_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

# %%
# Load environment variables
default_values.load_environment_files()

# %%
# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()

# %%
# Note:
# The dataset sources can be found at:
# - HuggingFace source: https://huggingface.co/datasets/walledai/TDC23-RedTeaming

# %%
# Create the orchestrator with scorer without safe prompts included
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

# Fetch only unsafe prompts from tdc23_redteaming dataset
prompt_dataset = fetch_tdc23_redteaming_dataset()

# Use the first 8 examples for red teaming
prompt_list = prompt_dataset.prompts[:8]


# Send prompts using the orchestrator and capture responses
try:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
    if responses:
        await orchestrator.print_conversations()  # type: ignore
    else:
        print("No valid responses were received from the orchestrator.")
except Exception as e:
    print(f"An error occurred while sending prompts: {e}")

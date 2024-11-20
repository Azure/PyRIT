# %% [markdown]
# # LibrAI "Do Not Answer" Dataset Testing
#
# This notebook demonstrates the process of using the LibrAI "Do Not Answer" dataset to perform AI red teaming on a target language model (LLM).
# It involves fetching potentially harmful behavior prompts, sending them to the LLM, and analyzing the responses.
# The goal is to identify vulnerabilities, inappropriate responses, or weaknesses in the model's handling of sensitive prompts.

# %%
# Import necessary packages
from pyrit.datasets import fetch_librAI_do_not_answer_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

# %%
# Set up the target
prompt_target = TextTarget()

# %%
# Create the orchestrator
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

# %%
# Fetch the dataset and limit to 5 prompts
prompt_dataset = fetch_librAI_do_not_answer_dataset()
prompt_list = [seed_prompt.value for seed_prompt in prompt_dataset.prompts[:5]]  # Extract values (text prompts)

# %%
# Send prompts using the orchestrator and capture responses
responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
if responses:
    await orchestrator.print_conversations()  # type: ignore

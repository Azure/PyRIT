# %% [markdown]
# # Red Team Social Bias dataset testing
#
# This dataset aggregates and unifies existing red-teaming prompts
# designed to identify stereotypes, discrimination, hate speech, and
# other representation harms in text-based Large Language Models (LLMs).


# %%
# Import necessary packages
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.datasets import fetch_red_team_social_bias_prompts_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget


# %%
# Load environment variables
initialize_pyrit(memory_db_type=IN_MEMORY)


# %%
# Set up the Azure OpenAI prompt target
prompt_target = OpenAIChatTarget()
examples_source = "svannie678/red_team_repo_social_bias_prompts"


# %%
# Note:
# The dataset sources can be found at:
# - GitHub repository: https://svannie678.github.io/svannie678-red_team_repo_social_bias/ 

# %%
# Create the orchestrator with scorer
orchestrator = PromptSendingOrchestrator(objective_target=prompt_target)

# Fetch examples from Red Team Social Bias dataset
dataset = fetch_red_team_social_bias_prompts_dataset()
prompt_value = [
    prompt.value for prompt in dataset.prompts
    
]

# Use the first 3 examples for red teaming
prompts_to_send = prompt_value[:3]

# Send prompts using the orchestrator and capture responses
try:
    responses = await orchestrator.send_prompts_async(prompt_list=prompts_to_send)  # type: ignore
    if responses:
        await orchestrator.print_conversations_async()  # type: ignore
    else:
        print("No valid responses were received from the orchestrator.")
except Exception as e:
    print(f"An error occurred while sending prompts: {e}")


# %%

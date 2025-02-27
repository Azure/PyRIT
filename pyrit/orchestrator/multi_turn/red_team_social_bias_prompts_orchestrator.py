# %% [markdown]
# # Red Team Social Bias Dataset Testing
#
# This dataset contains aggregated and unified existing red-teaming prompts designed to identify stereotypes,
# discrimination, hate speech, and other representation harms in text-based Large Language Models (LLMs)

# %%
# Import necessary packages
from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.datasets import fetch_red_team_social_bias_prompts_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget


initialize_pyrit(memory_db_type=IN_MEMORY)

# Set up the target
prompt_target = TextTarget()

# Fetch the dataset and limit to 8 prompts
prompt_dataset = fetch_red_team_social_bias_prompts_dataset()
orchestrator = PromptSendingOrchestrator(objective_target=prompt_target)
seed_prompt_group_list = prompt_dataset.group_seed_prompts_by_prompt_group_id(prompt_dataset.prompts)

all_responses = []

# Send prompts using the orchestrator and capture responses
for seed_prompt_group in seed_prompt_group_list[:8]:
    prompt_values = [seed_prompt.value for seed_prompt in seed_prompt_group.prompts]
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_values)
    all_responses.extend(responses)

for response in all_responses:
    await response.print_conversations_async()

orchestrator.dispose_db_engine()

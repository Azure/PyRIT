# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: pyrit
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Querying by Harm Categories
#
# This notebook shows how you can query attack results by harm category, as this data is not duplicated into the attack results. Instead we can use SQL queries to do this filtering.

# %% [markdown]
# ## Import Seed Prompt Dataset
#
# First we import a dataset which has individual prompts with differnt harm categories as an example.

# %%
import pathlib

from pyrit.common.initialization import initialize_pyrit
from pyrit.common.path import DATASETS_PATH
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import SeedPromptDataset

initialize_pyrit(memory_db_type="InMemory")

memory = CentralMemory.get_memory_instance()

seed_prompts = SeedPromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "seed_prompts" / "illegal.prompt")

print(f"Dataset name: {seed_prompts.dataset_name}")
print(f"Number of prompts in dataset: {len(seed_prompts.prompts)}")
print()

await memory.add_seed_prompts_to_memory_async(prompts=seed_prompts.prompts, added_by="bolor")  # type: ignore
for i, prompt in enumerate(seed_prompts.prompts):
    print(f"Prompt {i+1}: {prompt.value}, Harm Categories: {prompt.harm_categories}")

# %% [markdown]
# ## Send to target
#
# We use prompt sending attack to create our attack results

# %%
from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import OpenAIChatTarget

# Create a real OpenAI target
target = OpenAIChatTarget()

# Create the attack with the OpenAI target
attack = PromptSendingAttack(objective_target=target)

# Get our seed prompt groups with harm categories
groups = memory.get_seed_prompt_groups()
print(f"Total groups: {len(groups)}")

# Configure this to load the prompts loaded in the previous step.
# In the last section, they were in the illegal.prompt file (which has a configured name of "2025_06_pyrit_illegal_example")
prompt_groups = memory.get_seed_prompt_groups(dataset_name="2025_06_pyrit_illegal_example")
print(f"Found {len(prompt_groups)} prompt groups for dataset")

for i, group in enumerate(prompt_groups):
    prompt = group.prompts[0].value

    results = await attack.execute_async(objective=prompt, seed_prompt_group=group)

    print(f"Attack completed - Conversation ID: {results.conversation_id}")
    await ConsoleAttackResultPrinter().print_conversation_async(result=results)  # type: ignore

# %% [markdown]
# ## Query by harm category
# Now you can query your attack results by harm_category!

# %% [markdown]
# ### Single harm category:
#
# Query by a single harm category, as example here we query for `illegal`

# %%
# Demonstrating how to query attack results by harm category
print("=== Querying Attack Results by Harm Category ===")
print()

# First, let's see all attack results to understand what we have
all_attack_results = memory.get_attack_results()
print(f"Total attack results in memory: {len(all_attack_results)}")
print()

# Example 1: Query for a single harm category
print("1. Query for single harm category 'illegal':")
illegal_attacks = memory.get_attack_results(harm_category=["illegal"])
print(f"   Found {len(illegal_attacks)} attack results with 'illegal' category")

for i, attack in enumerate(illegal_attacks):  # Show first 2
    print(f"   Attack {i+1}: {attack.objective}...")
    print(f"   Conversation ID: {attack.conversation_id}")
print()

# %% [markdown]
# ### Query by multiple harm categories

# %%
# Example 2: Query for multiple harm categories
print("3. Query for multiple harm categories 'illegal' and 'violence':")
multiple_groups_and = memory.get_attack_results(harm_category=["illegal", "violence"])

for i, attack in enumerate(multiple_groups_and):
    print(f"   Attack {i+1}: {attack.objective}...")
    print(f"   Conversation ID: {attack.conversation_id}")
print()

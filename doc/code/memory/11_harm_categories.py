# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Querying by Harm Categories
#
# This notebook demonstrates how to retrieve attack results based on harm category. While harm category information is not duplicated into the `AttackResultEntries` table, PyRIT provides functions that perform the necessary SQL queries to filter `AttackResults` by harm category.

# %% [markdown]
# ## Import Seed Dataset
#
# First we import a dataset which has individual prompts with different harm categories as an example.

# %%
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import SeedDataset
from pyrit.setup.initialization import initialize_pyrit

initialize_pyrit(memory_db_type="InMemory")

memory = CentralMemory.get_memory_instance()

seed_prompts = SeedDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "seed_prompts" / "airt" / "illegal.prompt")

print(f"Dataset name: {seed_prompts.dataset_name}")
print(f"Number of prompts in dataset: {len(seed_prompts.prompts)}")
print()

await memory.add_seeds_to_memory_async(prompts=seed_prompts.prompts, added_by="bolor")  # type: ignore
for i, prompt in enumerate(seed_prompts.prompts):
    print(f"Prompt {i+1}: {prompt.value}, Harm Categories: {prompt.harm_categories}")

# %% [markdown]
# ## Send to target
#
# We use `PromptSendingAttack` to create our `AttackResults`

# %%
from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import OpenAIChatTarget

# Create a real OpenAI target
target = OpenAIChatTarget()

# Create the attack with the OpenAI target
attack = PromptSendingAttack(objective_target=target)

# Configure this to load the prompts loaded in the previous step.
# In the last section, they were in the illegal.prompt file (which has a configured name of "2025_06_pyrit_illegal_example")
prompt_groups = memory.get_seed_groups(dataset_name="2025_06_pyrit_illegal_example")
print(f"Found {len(prompt_groups)} prompt groups for dataset")

for i, group in enumerate(prompt_groups):
    prompt_text = group.prompts[0].value

    results = await attack.execute_async(objective=prompt_text, seed_group=group)  # type: ignore

    print(f"Attack completed - Conversation ID: {results.conversation_id}")
    await ConsoleAttackResultPrinter().print_conversation_async(result=results)  # type: ignore

# %% [markdown]
# ## Query by harm category
# Now you can query your attack results by `targeted_harm_category`!

# %% [markdown]
# ### Single harm category:
#
# Here, we by a single harm category (eg shown below is querying for the harm category  `['illegal']`)

# %%
from pyrit.analytics.result_analysis import analyze_results

all_attack_results = memory.get_attack_results()

# Demonstrating how to query attack results by harm category
print("=== Querying Attack Results by Harm Category ===")
print()

# First, let's see all attack results to understand what we have
print(f"Overall attack analytics:")
print(f"Total attack results in memory: {len(all_attack_results)}")

overall_analytics = analyze_results(list(all_attack_results))

# Access the Overall stats
overall_stats = overall_analytics["Overall"]
print(f"  Success rate: {overall_stats.success_rate}")
print(f"  Successes: {overall_stats.successes}")
print(f"  Failures: {overall_stats.failures}")
print(f"  Undetermined: {overall_stats.undetermined}")
print()

# Example 1: Query for a single harm category
print("1. Query for single harm category 'illegal':")
illegal_attacks = memory.get_attack_results(targeted_harm_categories=["illegal"])
print(f"\tFound {len(illegal_attacks)} attack results with 'illegal' category")

if illegal_attacks:
    for i, attack_result in enumerate(illegal_attacks):
        print(f"Attack {i+1}: {attack_result.objective}")
        print(f"Conversation ID: {attack_result.conversation_id}")
        print(f"Outcome: {attack_result.outcome}")
    print()

# %% [markdown]
# ### Multiple harm categories:

# %%
# Example 2: Query for multiple harm categories
print("2. Query for multiple harm categories 'illegal' and 'violence':")
multiple_groups = memory.get_attack_results(targeted_harm_categories=["illegal", "violence"])

for i, attack_result in enumerate(multiple_groups):
    print(f"Attack {i+1}: {attack_result.objective}...")
    print(f"Conversation ID: {attack_result.conversation_id}")
print()

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 8. Seed Prompt Database
#
# Apart from storing results in memory it's also useful to store datasets of seed prompts
# and seed prompt templates that we may want to use at a later point.
# This can help us in curating prompts with custom metadata like harm categories.
# As with all memory, we can use local DuckDBMemory or AzureSQLMemory in Azure to get the
# benefits of sharing with other users and persisting data.

# %%

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory

default_values.load_environment_files()

azure_memory = AzureSQLMemory()


# %% [markdown]
# ## Adding prompts to the database

# %%
from pyrit.models import SeedPromptDataset
from pyrit.common.path import DATASETS_PATH
import pathlib

seed_prompt_dataset = SeedPromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "seed_prompts" / "illegal.prompt")

print(seed_prompt_dataset.prompts[0])

azure_memory.add_seed_prompts_to_memory(prompts=seed_prompt_dataset.prompts, added_by="test")

# %% [markdown]
# ## Retrieving prompts from the database
#
# First, let's get an idea of what datasets are represented in the database.

# %%
azure_memory.get_seed_prompt_dataset_names()

# %% [markdown]
# The dataset we just uploaded (called "test illegal") is also represented.
# To get all seed prompts from that dataset, we can query as follows:

# %%
dataset_name = "test illegal"
prompts = azure_memory.get_seed_prompts(dataset_name=dataset_name)
print(f"Total number of the prompts with dataset name '{dataset_name}':", len(prompts))
print(prompts[0].__dict__)

# %% [markdown]
# # Adding seed prompt groups to the database
# %%
from pyrit.models import SeedPromptGroup

seed_prompt_group = SeedPromptGroup.from_yaml_file(
    pathlib.Path(DATASETS_PATH) / "seed_prompts" / "illegal-multimodal.prompt"
)

azure_memory.add_seed_prompt_groups_to_memory(prompt_groups=[seed_prompt_group], added_by="test multimodal illegal")

# %% [markdown]
# ## Retrieving seed prompt groups from the memory with dataset_name as "test multimodal"

# %%

multimodal_dataset_name = "test multimodal"
seed_prompt_groups = azure_memory.get_seed_prompt_groups(dataset_name=multimodal_dataset_name)
print(f"Total number of the seed prompt groups with dataset name '{multimodal_dataset_name}':", len(seed_prompt_groups))
print(seed_prompt_groups[0].__dict__)

# %%

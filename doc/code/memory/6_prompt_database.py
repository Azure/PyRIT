# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: pyrit-kernel
#     language: python
#     name: pyrit-kernel
# ---

# %% [markdown]
# ## Prompt Database in Azure
#
# With AzureSQLMemory we have a centralized database in Azure. Apart from storing results it's also useful to store datasets of prompts and prompt templates that we may want to use at a later point. This can help us in curating prompts with custom metadata like harm categories.

# %%
import os

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory

default_values.load_default_env()

azure_memory = AzureSQLMemory()


# %% [markdown]
# ## Adding prompts to the database

# %%
from pyrit.models import PromptDataset
from pyrit.common.path import DATASETS_PATH
import pathlib

prompt_dataset = PromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "prompts" / "illegal.prompt")

print(prompt_dataset.prompts[0])

azure_memory.add_prompts_to_memory(prompts=prompt_dataset.prompts, added_by="test")

# %% [markdown]
# # Retrieving prompts from the database
#
# First, let's get an idea of what datasets are represented in the database.

# %%
azure_memory.get_prompt_dataset_names()

# %% [markdown]
# The dataset we just uploaded (called "test illegal") is also represented. To get all prompts from that dataset, we can query as follows:

# %%
prompts = azure_memory.get_prompts(dataset_name="test illegal")
print(prompts[0].__dict__)

# %%

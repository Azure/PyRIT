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
# TODO context

# %%
import os

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory

default_values.load_default_env()

azure_memory = AzureSQLMemory()


# %%
import os
os.environ

# %%
azure_memory.


# %%

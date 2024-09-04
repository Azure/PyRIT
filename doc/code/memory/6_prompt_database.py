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
# from pyrit.memory import AzureSQLMemory
from pyrit.memory import DuckDBMemory
default_values.load_default_env()

#conn_str = os.environ.get('AZURE_SQL_SERVER_CONNECTION_STRING')

#azure_memory = AzureSQLMemory(connection_string=conn_str)
memory = DuckDBMemory()

# %%
memory

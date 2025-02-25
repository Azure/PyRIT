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
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. DuckDB Memory
#
# The memory DuckDB database can be thought of as a normalized source of truth. The memory module is the primary way pyrit keeps track of requests and responses to targets and scores. Most of this is done automatically. All Prompt Targets write to memory for later retrieval. All scorers also write to memory when scoring.
#
# The schema is found in `memory_models.py` and can be programatically viewed as follows

# %%
from pyrit.memory import DuckDBMemory

memory = DuckDBMemory()
memory.print_schema()

memory.dispose_engine()


# %% [markdown]
#

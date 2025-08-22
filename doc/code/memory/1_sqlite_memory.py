# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # 1. SQLite Memory
#
# The memory SQLite database can be thought of as a normalized source of truth. The memory module is the primary way PyRIT keeps track of requests and responses to targets and scores. Most of this is done automatically. All Prompt Targets write to memory for later retrieval. All scorers also write to memory when scoring.
#
# The schema is found in `memory_models.py` and can be programatically viewed as follows

# %%
from pyrit.memory import SQLiteMemory

# Use in-memory database to avoid file corruption issues
memory = SQLiteMemory(db_path=":memory:")
memory.print_schema()

memory.dispose_engine()


# %% [markdown]
# ## DuckDB to SQLite Migration
#
# **Note:** DuckDBMemory has been replaced by SQLiteMemory for better compatibility and performance.
#
# If you try to use DuckDBMemory, it will:
# 1. Issue a warning
# 2. Use in-memory SQLite instead

# %%
# Example: What happens when you try to use DuckDB
# This demonstrates the helpful error message that guides users to migrate

from pyrit.common.initialization import initialize_pyrit

try:
    # Attempting to initialize with DuckDB will show a helpful migration message
    initialize_pyrit(memory_db_type="DuckDB")

except Exception as e:
    print("Error:", e)

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: pyrit-311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Memory
#
# The memory DuckDB database can be thought of as a normalized source of truth. The memory module is the primary way pyrit keeps track of requests and responses to targets and scores. Most of this is done automatically. All Prompt Targets write to memory for later retrieval. All scorers also write to memory when scoring.
#
# The schema is found in `memory_models.py` and can be programatically viewed as follows

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory import DuckDBMemory

memory = DuckDBMemory()
memory.print_schema()

# %% [markdown]
# ## Basic Programming Usage
#
# The `pyrit.memory` module provides functionality to keep track of the conversation history, scoring, data, and more. You can use memory to read and write data. Here is an example that retrieves a normalized conversation:

# %%


from uuid import uuid4
from pyrit.models import PromptRequestPiece, PromptRequestResponse

conversation_id = str(uuid4())

message_list = [
    PromptRequestPiece(
        role="user", original_value="Hi, chat bot! This is my initial prompt.", conversation_id=conversation_id
    ),
    PromptRequestPiece(
        role="assistant", original_value="Nice to meet you! This is my response.", conversation_id=conversation_id
    ),
    PromptRequestPiece(
        role="user",
        original_value="Wonderful! This is my second prompt to the chat bot!",
        conversation_id=conversation_id,
    ),
]

memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[0]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[1]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[2]]))


entries = memory._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)

for entry in entries:
    print(entry)


# Cleanup memory resources
memory.dispose_engine()


# %% [markdown]
# ## Updating Memory Manually
#
#
# After or during an operation or a test, it can be important to use the memory in the database in a straightforward way.
#
# There are many ways to do this, but this section gives some general ideas on how users can solve common problems. Most of this relies on using https://duckdb.org/docs/guides/sql_editors/dbeaver.html
#
#
# ### Sharing Data Between Users
#
# Eventually, we have plans to extend the `MemoryInterface` implementation to other instances. For example, it would not be a huge task to extend it to Azure SQL Server, and operators could use that as a shared database.
#
# In the mean time, one of the easiest ways to share data is to do the following:
#
# 1. Export and import the database as described here. This allows a lot of flexibility and can include partial exports (for example based on labels or time):  https://dbeaver.com/docs/dbeaver/Data-transfer/
# 2. Copy the PyRIT `results/dbdata` directory over; it will contain multi-modal data that the database references.
#
# ### Making Pretty Graphs with Excel
#
# This is especially nice with scoring. There are countless ways to do this, but this shows one example;
#
# 1. Do a query with the data you want. This is an example where we're only quering scores in the "float_type" scores in the category of "misinformation"
#
# ![scoring_1.png](../../../assets/scoring_1.png)
#
# 1. Export the data to a CSV
#
# ![scoring_2.png](../../../assets/scoring_2_export.png)
#
# 1. Use it as an excel sheet! You can use pivot tables, etc. To visualize the data.
#
# ![scoring_2.png](../../../assets/scoring_3_pivot.png)
#
# 1. Optionally, if you caught things you wanted to update, you could either change in the database or directly in the excel and re-import. Note: mapping is sometimes not 100% so updating in the database is best.
#
# ### Entering Manual Prompts
#
# Although most prompts are run through `PromptTargets` which will add prompts to memory, there are a few reasons you may want to enter in manual prompts. For example, if you ssh into a box, are not using PyRIT to probe for weaknesses, but want to add prompts later for reporting or scoring.
#
# One of the easiest way to add prompts is through the `TextTarget` target. You can create a csv of prompts that looks as follows:
#
# ```
# role, value
# user, hello
# assistant, hi how are you?
# user, new conversation
# ```
#
# This very simple format doesn't have very much information, but already it standardizes the prompts that can then be used in mass scoring (or manuual scoring with HITLScorer).
#
# And you can import it using code like this
#
# ```
# target = TextTarget()
# target.import_scores_from_csv(csv_file_path=".\path.csv")
# ```
#
#
#

# %% [markdown]
#

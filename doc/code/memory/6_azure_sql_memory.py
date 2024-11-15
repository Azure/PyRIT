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
#     display_name: pyrit-kernel
#     language: python
#     name: pyrit-kernel
# ---

# %% [markdown]
# # 6. Azure SQL Memory
#
# The memory AzureSQL database can be thought of as a normalized source of truth. The memory module is the primary way pyrit keeps track of requests and responses to targets and scores. Most of this is done automatically. All orchestrators write to memory for later retrieval. All scorers also write to memory when scoring.
#
# The schema is found in `memory_models.py` and can be programatically viewed as follows
#
# ## Azure Login
#
# PyRIT `AzureSQLMemory` supports only **Azure Entra ID authentication** at this time. User ID/password-based login is not available.
#
# Please log in to your Azure account before running this notebook:
#
# - Log in with the proper scope to obtain the correct access token:
#   ```bash
#   az login --scope https://database.windows.net//.default
#   ```
# ## Environment Variables
#
# Please set the following environment variables to run AzureSQLMemory interactions:
#
# - `AZURE_SQL_DB_CONNECTION_STRING` = "<Azure SQL DB connection string here in SQLAlchemy format>"
# - `AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL` = "<Azure Storage Account results container URL>" (which uses delegation SAS) but needs login to Azure.
#
# To use regular key-based authentication, please also set:
#
# - `AZURE_STORAGE_ACCOUNT_RESULTS_SAS_TOKEN`
#

# %%
from pyrit.memory import AzureSQLMemory


memory = AzureSQLMemory()

memory.print_schema()


# %% [markdown]
# ## Basic Azure SQL Memory Programming Usage
#
# The `pyrit.memory.azure_sql_memory` module provides functionality to keep track of the conversation history, scoring, data, and more using Azure SQL. You can use memory to read and write data. Here is an example that retrieves a normalized conversation:

# %%
from uuid import uuid4
from pyrit.memory import AzureSQLMemory
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

memory = AzureSQLMemory()

memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[0]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[1]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[2]]))


entries = memory.get_conversation(conversation_id=conversation_id)

for entry in entries:
    print(entry)


# Cleanup memory resources
memory.dispose_engine()

# %%

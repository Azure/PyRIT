# %% [markdown]
# The `pyrit.memory` module provides functionality to dump the database tables into JSON files, creating one file per table. Also, dump the data for a given conversation id. In a nutshell, this can be used as follows

# %% [markdown]
# ### Export all DuckDB tables to JSON format files
#
# This will export all DuckDB tables to JSON files, which will be stored in the `pyrit/results` folder.

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory import DuckDBMemory
from uuid import uuid4
from pyrit.models import PromptRequestPiece

duckdb_memory = DuckDBMemory()
duckdb_memory.export_all_tables()

# %% [markdown]
# ### Export Conversation Data to JSON for a Conversation ID
# This functionality exports all conversation data associated with a specific conversation ID to a JSON file. The file, named using the format `conversation_id.json`, will be located in the `pyrit/results` folder.

# %%
conversation_id = str(uuid4())

message_list = [
    PromptRequestPiece(
        role="user", original_prompt_text="Hi, chat bot! This is my initial prompt.", conversation_id=conversation_id
    ),
    PromptRequestPiece(
        role="assistant", original_prompt_text="Nice to meet you! This is my response.", conversation_id=conversation_id
    ),
]

next_message = PromptRequestPiece(
    role="user",
    original_prompt_text="Wonderful! This is my second prompt to the chat bot.",
    conversation_id=conversation_id,
)

message_list.append(next_message)
duckdb_memory.add_request_pieces_to_memory(request_pieces=message_list)

# %%
duckdb_memory.export_conversation_by_id(conversation_id=conversation_id)

# %%
# Cleanup DuckDB resources
duckdb_memory.dispose_engine()

# %%

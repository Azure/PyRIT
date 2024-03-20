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
from pyrit.models import ChatMessage

duckdb_memory = DuckDBMemory()
duckdb_memory.export_all_tables()

# %% [markdown]
# ### Export Conversation Data to JSON for a Conversation ID
# This functionality exports all conversation data associated with a specific conversation ID to a JSON file. The file, named using the format `conversation_id.json`, will be located in the `pyrit/results` folder.

# %%
conversation_id = str(uuid4())

message_list = [
    ChatMessage(role="user", content="Hi, chat bot! This is my initial prompt."),
    ChatMessage(role="assistant", content="Nice to meet you! This is my response."),
]
next_message = ChatMessage(role="user", content="Wonderful! This is my second prompt to the chat bot.")
message_list.append(next_message)
duckdb_memory.add_chat_messages_to_memory(conversations=message_list, conversation_id=conversation_id)

# %%
duckdb_memory.export_conversation_by_id(conversation_id=conversation_id)

# %%
# Cleanup DuckDB resources
duckdb_memory.dispose_engine()

# %%

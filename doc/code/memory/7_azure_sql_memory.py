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
#     name: pyrit-311
# ---

# %% [markdown]
# ## Azure SQL Server Memory Usage
#
# The `pyrit.memory` module provides functionality via Azure SQL Server to keep track of the conversation history, scoring, data, and more. You can use memory to read and write data. Here is an example that retrieves a normalized conversation:

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from uuid import uuid4
from pyrit.common import default_values
from pyrit.memory.azure_sql_memory import AzureSQLMemory
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

default_values.load_default_env()

memory = AzureSQLMemory(connection_string=os.environ.get("AZURE_SQL_SERVER_CONNECTION_STRING"))

memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[0]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[1]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[2]]))


entries = memory._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)

for entry in entries:
    print(entry)


# Cleanup memory resources
memory.dispose_engine()

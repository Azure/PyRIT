# %% [markdown]
# ## Basic Memory Programming Usage
#
# The `pyrit.memory` module provides functionality to keep track of the conversation history, scoring, data, and more. You can use memory to read and write data. Here is an example that retrieves a normalized conversation:

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from uuid import uuid4
from pyrit import memory
from pyrit.memory.duckdb_memory import DuckDBMemory
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

memory = DuckDBMemory()

memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[0]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[1]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[2]]))


entries = memory._get_prompt_pieces_with_conversation_id(conversation_id=conversation_id)

for entry in entries:
    print(entry)


# Cleanup memory resources
memory.dispose_engine()
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %% [markdown]
# # 2. Basic Memory Programming Usage
#
# The `pyrit.memory` module provides functionality to keep track of the conversation history, scoring, data, and more. You can use memory to read and write data. Here is an example that retrieves a normalized conversation using SQLite memory:

# %%
from uuid import uuid4

from pyrit.memory import SQLiteMemory
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

memory = SQLiteMemory()

memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[0]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[1]]))
memory.add_request_response_to_memory(request=PromptRequestResponse([message_list[2]]))

entries = memory.get_conversation(conversation_id=conversation_id)

for entry in entries:
    print(entry)

# Cleanup memory resources
memory.dispose_engine()
